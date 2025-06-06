import argparse
import time
import numpy as np
import data
import models
import os
import utils
from datetime import datetime
np.set_printoptions(precision=5, suppress=True)
from scipy.stats import binom
from tqdm import tqdm
import torchvision
import cv2

import videoseal
from videoseal.evals.metrics import bit_accuracy
from videoseal.utils.cfg import setup_dataset, setup_model_from_checkpoint, setup_model_from_model_card
import torch

import yaml
import model
from easydict import EasyDict

from REVMark import Encoder, Decoder, framenorm


def find_videos(folder_path):
    video_extensions = ['.mp4']
    video_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    return video_files


class video_dataset(torch.utils.data.Dataset):
    def __init__(self, video_path):
        self.video_files = []
        for video_file in os.listdir(video_path):
            if video_file.endswith('.mp4'):
                self.video_files.append(os.path.join(video_path, video_file))
        self.video_files.sort()

    def resize_frame(self, frame):
        return cv2.resize(frame, (512, 512))

    def load_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        video = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.resize_frame(frame)
            video.append(torch.from_numpy(frame.transpose(2, 0, 1).astype('float32') / 255))

        return torch.stack(video, dim=1)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video = self.load_video(video_file)
        video = video.permute(1, 0, 2, 3)
        video_name = video_file.split('/')[-1]
        return video, video_name


class video_dataset_revmark(torch.utils.data.Dataset):
    def __init__(self, video_path):
        self.video_files = []
        for video_file in os.listdir(video_path):
            if video_file.endswith('.mp4'):
                self.video_files.append(os.path.join(video_path, video_file))
        self.video_files.sort()

    def resize_frame(self, frame):
        return cv2.resize(frame, (128, 128))

    def load_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        video = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video.append(torch.from_numpy(frame[:128, :128].transpose(2, 0, 1).astype('float32') / 255))

        return torch.stack(video, dim=1)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video = self.load_video(video_file)
        video = self.reshape_tensor(video)
        video_name = video_file.split('/')[-1]
        return video, video_name

    def reshape_tensor(self, tensor):
        c, y, h, w = tensor.shape
        x = (y + 7) // 8
        pad_size = x * 8 - y
        if pad_size > 0:
            tensor = torch.cat([tensor, tensor[:, :pad_size]], dim=1)
        tensor = tensor.view(c, x, 8, h, w)
        tensor = tensor.permute(1, 0, 2, 3, 4)
        return tensor


def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p


def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
              max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def square_attack_linf_revmark(model, x, y, corr_classified, eps, n_iters, p_init, metrics_path, targeted, loss_type, dataset, tau, device):
    logit_thresh = np.log(tau / (1 - tau))
    B, F, C, K, H, W = x.shape
    x = x.reshape(B, F*C*K, H, W)
    x = x.detach().cpu().numpy()

    """ The Linf square attack """
    np.random.seed(0)
    min_val, max_val = 0, 1 if x.max() <= 1 else 255
    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]
    x, y = x[corr_classified], y[corr_classified]

    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    x_best = np.clip(x + init_delta, min_val, max_val)

    x_best = torch.from_numpy(x_best)
    x_best = x_best.reshape(x_best.shape[0], F, C, K, H, W)
    logits = model.detect(x_best.to(device).float())
    x_best = x_best.reshape(x_best.shape[0], F * C * K, H, W)
    x_best = x_best.numpy()
    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    n_queries = np.ones(x.shape[0])

    time_start = time.time()
    metrics = np.zeros([n_iters, 7])
    acc_list = []
    for i_iter in range(n_iters - 1):
        idx_to_fool = (logits[:,1] >= logit_thresh)
        x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])

        x_new = np.clip(x_curr + deltas, min_val, max_val)

        x_new = torch.from_numpy(x_new)
        x_new = x_new.reshape(x_new.shape[0], F, C, K, H, W)
        logits = model.detect(x_new.to(device).float())
        x_new = x_new.reshape(x_new.shape[0], F*C*K, H, W)
        x_new = x_new.numpy()

        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        x_best = torch.from_numpy(x_best)
        x_best = x_best.reshape(x_best.shape[0], F, C, K, H, W)
        logits = model.detect(x_best.to(device).float())
        x_best = x_best.reshape(x_best.shape[0], F * C * K, H, W)
        x_best = x_best.numpy()

        acc = (logits[:,1] >= logit_thresh).sum() / n_ex_total
        acc_corr = (logits[:,1] >= logit_thresh).mean()
        mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[logits[:,1] < logit_thresh]), np.median(n_queries[logits[:,1] < logit_thresh])
        avg_margin_min = np.mean(margin_min)
        time_total = time.time() - time_start
        log.print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
            format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], eps, time_total))

    return n_queries, x_best.reshape(x_best.shape[0], F, C, K, H, W)


def square_attack_linf_stegastamp(model, x, y, corr_classified, eps, n_iters, p_init, metrics_path, targeted, loss_type, dataset, tau, device):
    logit_thresh = np.log(tau / (1 - tau))
    B, F, C, H, W = x.shape
    x = x.reshape(B, F*C, H, W)
    x = x.cpu().numpy()

    """ The Linf square attack """
    np.random.seed(0)
    min_val, max_val = 0, 1 if x.max() <= 1 else 255
    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]
    x, y = x[corr_classified], y[corr_classified]

    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    x_best = np.clip(x + init_delta, min_val, max_val)

    x_best = torch.from_numpy(x_best)
    x_best = x_best.reshape(x_best.shape[0], F, C, H, W)
    logits = model.detect(x_best.to(device).float())
    x_best = x_best.reshape(x_best.shape[0], F * C, H, W)
    x_best = x_best.numpy()
    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    n_queries = np.ones(x.shape[0])

    time_start = time.time()
    metrics = np.zeros([n_iters, 7])
    acc_list = []
    for i_iter in range(n_iters - 1):
        idx_to_fool = (logits[:,1] >= logit_thresh)
        x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])

        x_new = np.clip(x_curr + deltas, min_val, max_val)

        x_new = torch.from_numpy(x_new)
        x_new = x_new.reshape(x_new.shape[0], F, C, H, W)
        logits = model.detect(x_new.to(device).float())
        x_new = x_new.reshape(x_new.shape[0], F*C, H, W)
        x_new = x_new.numpy()

        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        x_best = torch.from_numpy(x_best)
        x_best = x_best.reshape(x_best.shape[0], F, C, H, W)
        logits = model.detect(x_best.to(device).float())
        print(logits)
        x_best = x_best.reshape(x_best.shape[0], F * C, H, W)
        x_best = x_best.numpy()

        acc = (logits[:,1] >= logit_thresh).sum() / n_ex_total
        acc_corr = (logits[:,1] >= logit_thresh).mean()
        mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[logits[:,1] < logit_thresh]), np.median(n_queries[logits[:,1] < logit_thresh])
        avg_margin_min = np.mean(margin_min)
        time_total = time.time() - time_start
        log.print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
            format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], eps, time_total))

    return n_queries, x_best.reshape(x_best.shape[0], F, C, H, W)


def square_attack_linf(model, x, y, corr_classified, eps, n_iters, p_init, metrics_path, targeted, loss_type, dataset, tau, device, aggregation):
    logit_thresh = np.log(tau / (1 - tau))
    B, F, C, H, W = x.shape
    x = x.reshape(B, F*C, H, W)
    x = x.cpu().numpy()

    """ The Linf square attack """
    np.random.seed(0)
    min_val, max_val = 0, 1 if x.max() <= 1 else 255
    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]
    x, y = x[corr_classified], y[corr_classified]

    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    x_best = np.clip(x + init_delta, min_val, max_val)

    x_best = torch.from_numpy(x_best)
    x_best = x_best.reshape(x_best.shape[0], F, C, H, W)
    logits = model.detect(x_best.float(), aggregation)
    x_best = x_best.reshape(x_best.shape[0], F * C, H, W)
    x_best = x_best.numpy()
    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    n_queries = np.ones(x.shape[0])

    time_start = time.time()
    metrics = np.zeros([n_iters, 7])
    acc_list = []
    for i_iter in range(n_iters - 1):
        idx_to_fool = (logits[:,1] >= logit_thresh)
        x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])

        x_new = np.clip(x_curr + deltas, min_val, max_val)

        x_new = torch.from_numpy(x_new)
        x_new = x_new.reshape(x_new.shape[0], F, C, H, W)
        logits = model.detect(x_new.float(), aggregation)
        x_new = x_new.reshape(x_new.shape[0], F*C, H, W)
        x_new = x_new.numpy()

        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        x_best = torch.from_numpy(x_best)
        x_best = x_best.reshape(x_best.shape[0], F, C, H, W)
        logits = model.detect(x_best.float(), aggregation)
        print(logits)
        x_best = x_best.reshape(x_best.shape[0], F * C, H, W)
        x_best = x_best.numpy()

        acc = (logits[:,1] >= logit_thresh).sum() / n_ex_total
        acc_corr = (logits[:,1] >= logit_thresh).mean()
        mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[logits[:,1] < logit_thresh]), np.median(n_queries[logits[:,1] < logit_thresh])
        avg_margin_min = np.mean(margin_min)
        time_total = time.time() - time_start
        log.print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
            format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], eps, time_total))

    return n_queries, x_best.reshape(x_best.shape[0], F, C, H, W)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model', type=str, default='videoseal', help='Model name.')
    parser.add_argument('--attack', type=str, default='square_linf', choices=['square_linf', 'square_l2'], help='Attack.')
    parser.add_argument('--exp_folder', type=str, default='exps', help='Experiment folder to store all output.')
    parser.add_argument('--gpu', type=str, default='10', help='GPU number. Multiple GPUs are possible for PT models.')
    parser.add_argument('--n_ex', type=int, default=10000, help='Number of test ex to test on.')
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of changing a coordinate. Note: check the paper for the best values.'
                             'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--eps', type=float, default=12.75, help='Radius of the Lp ball.')
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--targeted', action='store_true', help='Targeted or untargeted attack.')
    parser.add_argument('--dataset', type=str, default='stable-video_realistic')
    parser.add_argument('--aggregation', type=str, default='ba-mean')
    parser.add_argument('--num_test_video', type=int, default=40)
    args = parser.parse_args()
    args.loss = 'margin_loss' if not args.targeted else 'cross_entropy'

    timestamp = str(datetime.now())[:-7]
    hps_str = '{} model={} attack={} n_ex={} eps={} p={} n_iter={}'.format(
        timestamp, args.model, args.attack, args.n_ex, args.eps, args.p, args.n_iter)
    args.eps = args.eps / 255.0
    n_cls = 2
    square_attack = square_attack_linf if args.attack == 'square_linf' else square_attack_l2

    log_path = '{}/{}.log'.format(args.exp_folder, hps_str)
    metrics_path = '{}/{}.metrics'.format(args.exp_folder, hps_str)
    log = utils.Logger(log_path)
    log.print('All hps: {}'.format(hps_str))


    ### VideoSeal
    if args.model == 'videoseal':
        # Setup model
        model = setup_model_from_model_card('./videoseal/cards/videoseal.yaml')
        model.eval()
        model.blender.scaling_w = model.blender.scaling_w
        model.chunk_size = model.chunk_size
        model.step_size = model.step_size
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # Load video
        folder_path = './kinetics-dataset/'
        videos = find_videos(folder_path)[:args.num_test_video]
        print(videos)
        for video_path in tqdm(videos):
            video = torchvision.io.read_video(video_path, output_format="TCHW")
            video = video[0].float() / 255.0
            video = video.unsqueeze(0).to(device)
            video = video[:, :14, :, :, :]
            print(video.shape)

            if args.aggregation in ['ba-mean', 'ba-median', 'bit-median', 'logit-mean', 'logit-median']:
                tau = 29/96
            elif args.aggregation == 'detection-threshold':
                n = video.shape[1]
                p = 0.0000661
                threshold = 10e-4
                for i in range(n + 1):
                    if binom.sf(i - 1, n, p) <= threshold:
                        k = i
                        break
                tau = 1 - k / n
            elif args.aggregation == 'detection-median':
                tau = 0.5
            print(tau)
            logit_thresh = np.log(tau / (1 - tau))
            logits_clean = model.detect(video, args.aggregation)

            corr_classified = (logits_clean[:, 1] >= logit_thresh)
            print(f'logit_threshold: {logit_thresh}')
            log.print('Clean accuracy: {:.2%}'.format(np.mean(np.array(corr_classified))))

            y_test = np.ones(video.shape[0], dtype=np.int64)
            y_target = utils.random_classes_except_current(y_test, n_cls) if args.targeted else y_test
            y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)

            n_queries, x_adv = square_attack(model, video, y_target_onehot, corr_classified, args.eps, args.n_iter,
                                             args.p, metrics_path, args.targeted, args.loss, args.dataset, tau, device, args.aggregation)


    ### StegaStamp
    elif args.model == 'stegastamp':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open('eval.yaml', 'r') as f:
            cfg = EasyDict(yaml.safe_load(f))
        argss = EasyDict({**cfg})
        decoder = model.StegaStampDecoder(secret_size=argss.secret_size)
        decoder.load_state_dict(torch.load(os.path.join(argss.adv_pretrained, 'decoder.pth'), map_location=device))
        if argss.cuda:
            model = decoder.to(device)
        model.eval()

        dataset = video_dataset('./kinetics-dataset/')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        for i, (video, video_name) in enumerate(tqdm(dataloader)):
            if i >= args.num_test_video:
                break
            video = video.to(device)
            video = video[:, :14, :, :, :]
            print(video.shape)

            tau = 5 / 32
            print(tau)
            logit_thresh = np.log(tau / (1 - tau))
            logits_clean = model.detect(video)

            corr_classified = (logits_clean[:, 1] >= logit_thresh)
            print(f'logit_threshold: {logit_thresh}')
            log.print('Clean accuracy: {:.2%}'.format(np.mean(np.array(corr_classified))))

            y_test = np.ones(video.shape[0], dtype=np.int64)
            y_target = utils.random_classes_except_current(y_test, n_cls) if args.targeted else y_test
            y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)

            n_queries, x_adv, acc_list = square_attack_linf_stegastamp(model, video, y_target_onehot, corr_classified, args.eps, args.n_iter,
                                             args.p, metrics_path, args.targeted, args.loss, args.dataset, tau, device)


    ### REVMark
    elif args.model == 'revmark':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Decoder(96, [8, 128, 128]).to(device).eval()
        model.load_state_dict(torch.load('checkpoints/Decoder.pth'))
        model.tasblock.enable = True

        dataset = video_dataset_revmark('./kinetics-dataset/')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        acc = np.zeros(20)
        for i, (video, video_name) in enumerate(tqdm(dataloader)):
            if i >= 40:
                break
            video = video.to(device)
            video = video[:, :14, :, :, :]
            print(video.shape)

            tau = 29/96

            print(tau)
            logit_thresh = np.log(tau / (1 - tau))
            logits_clean = model.detect(video)

            corr_classified = (logits_clean[:, 1] >= logit_thresh)
            print(f'logit_threshold: {logit_thresh}')
            log.print('Clean accuracy: {:.2%}'.format(np.mean(np.array(corr_classified))))

            y_test = np.ones(video.shape[0], dtype=np.int64)
            y_target = utils.random_classes_except_current(y_test, n_cls) if args.targeted else y_test
            y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)

            n_queries, x_adv, acc_list = square_attack_linf_revmark(model, video, y_target_onehot, corr_classified, args.eps, args.n_iter,
                                             args.p, metrics_path, args.targeted, args.loss, args.dataset, tau, device)