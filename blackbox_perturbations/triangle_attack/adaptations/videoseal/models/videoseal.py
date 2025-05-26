# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ..augmentation.augmenter import Augmenter
from ..models.embedder import Embedder
from ..models.extractor import Extractor
from ..models.wam import Wam
from ..modules.jnd import JND

import numpy as np


class Videoseal(Wam):
    """
    A video watermarking model that extends the Wam class.
    This model combines an embedder, a detector, and an augmenter to embed watermarks into videos.
    It also includes optional attenuation and scaling parameters to control the strength of the watermark.
    Attributes:
        embedder (Embedder): The watermark embedder.
        detector (Extractor): The watermark detector.
        augmenter (Augmenter): The image augmenter.
        attenuation (JND, optional): The JND model to attenuate the watermark distortion. Defaults to None.
        scaling_w (float, optional): The scaling factor for the watermark. Defaults to 1.0.
        scaling_i (float, optional): The scaling factor for the image. Defaults to 1.0.
        chunk_size (int, optional): The number of frames/imgs to encode at a time. Defaults to 8.
        step_size (int, optional): The number of frames/imgs to propagate the watermark to. Defaults to 4.
        img_size (int, optional): The size of the images to resize to. Defaults to 256.
    """

    def __init__(
        self,
        embedder: Embedder,
        detector: Extractor,
        augmenter: Augmenter,
        attenuation: JND = None,
        scaling_w: float = 1.0,
        scaling_i: float = 1.0,
        img_size: int = 256,
        clamp: bool = True,
        chunk_size: int = 8,
        step_size: int = 4,
        blending_method: str = "additive"
    ) -> None:
        """
        Initializes the Videoseal model.
        Args:
            embedder (Embedder): The watermark embedder.
            detector (Extractor): The watermark detector.
            augmenter (Augmenter): The image augmenter.
            attenuation (JND, optional): The JND model to attenuate the watermark distortion. Defaults to None.
            scaling_w (float, optional): The scaling factor for the watermark. Defaults to 1.0.
            scaling_i (float, optional): The scaling factor for the image. Defaults to 1.0.
            img_size (int, optional): The size of the frame to resize to intermediately while generating the watermark then upscale, the final video / image size is kept the same. Defaults to 256.
            chunk_size (int, optional): The number of frames/imgs to encode at a time. Defaults to 8.
            step_size (int, optional): The number of frames/imgs to propagate the watermark to. Defaults to 4.
        """
        super().__init__(
            embedder=embedder,
            detector=detector,
            augmenter=augmenter,
            attenuation=attenuation,
            scaling_w=scaling_w,
            scaling_i=scaling_i,
            img_size=img_size,
            clamp=clamp,
            blending_method=blending_method
        )
        # video settings
        self.chunk_size = chunk_size  # embed 8 frames/imgs at a time
        self.step_size = step_size  # propagate the wm to 4 next frame/img

    def forward(
            self,
            imgs: torch.Tensor,
            aggregation,
            is_video: bool = True,
    ) -> dict:
        """
        Performs the forward pass of the detector only.
        Rescales the input images to 256x... pixels and then computes the mask and the message.
        Args:
            imgs (torch.Tensor): Batched images with shape FxCxHxW, where F is the number of frames,
                                    C is the number of channels, H is the height, and W is the width.
                                    if shape CxHxW is passed automatically will be considered as img
        Returns:
            dict: The output predictions.
                - torch.Tensor: Predictions for each frame with shape Fx(K+1),
                                where K is the length of the binary message. The first column represents
                                the probability of the detection bit, and the remaining columns represent
                                the probabilities of each bit in the message.
        """
        B, TC, H, W = imgs.shape
        T = TC // 3
        imgs = imgs.view(T, 3, H, W)

        if not is_video or len(imgs.shape) == 3:
            # fallback on parent class for batch of images
            return super().detect(imgs)
        all_preds = []
        for ii in range(0, len(imgs), self.chunk_size):
            nimgs_in_ck = min(self.chunk_size, len(imgs) - ii)
            outputs = super().detect(
                imgs[ii:ii + nimgs_in_ck]
            )
            preds = outputs["preds"]
            all_preds.append(preds)  # n k ..
        preds = torch.cat(all_preds, dim=0)  # f k ..
        outputs = {
            "preds": preds,  # predicted masks and/or messages: f (1+nbits) h w
        }

        logits = outputs["preds"][:, 1:]
        watermark_gt = np.array([0, 1] * 48)
        watermark_gt = np.tile(watermark_gt, (logits.shape[0], 1))

        ### BA-mean
        if aggregation == 'ba-mean':
            bitacc = np.mean((watermark_gt) == (logits.detach().cpu().numpy() >= 0))
            return bitacc >= 67 / 96

        ### BA-median
        elif aggregation == 'ba-median':
            bitacc = np.mean((logits.detach().cpu().numpy() >= 0) == watermark_gt, axis=1)
            bitacc = np.median(bitacc)
            return bitacc >= 67 / 96

        ### Logit-mean
        elif aggregation == 'logit-mean':
            decoded = np.mean(logits.detach().cpu().numpy(), axis=0)
            bitacc = np.sum((watermark_gt) == (decoded >= 0)) / (logits.shape[0] * logits.shape[1])
            return bitacc >= 67 / 96

        ### Logit-median
        elif aggregation == 'logit-median':
            from scipy.optimize import minimize
            def objective(median):
                return np.sum(np.linalg.norm(logits.detach().cpu().numpy() - median, axis=1))

            initial_guess = np.mean(logits.detach().cpu().numpy(), axis=0)
            decoded = minimize(objective, initial_guess, method='Powell', tol=1e-5)
            bitacc = np.sum((watermark_gt) == (decoded.x >= 0)) / (logits.shape[0] * logits.shape[1])
            return bitacc >= 67 / 96

        ### Bit-median
        elif aggregation == 'bit-median':
            decode = (logits.detach().cpu().numpy() >= 0)
            decoded = np.sum(decode, axis=0) > (len(decode) / 2)
            bitacc = np.sum((watermark_gt) == decoded) / (logits.shape[0] * logits.shape[1])
            return bitacc >= 67 / 96

        ### Detection-threshold & median
        elif aggregation == 'detection-threshold' or aggregation == 'detection-median':
            from scipy.stats import binom
            bitacc = np.mean((logits.detach().cpu().numpy() >= 0) == watermark_gt, axis=1)
            bitacc = np.sum(bitacc >= 67 / 96)
            if aggregation == 'detection-threshold':
                n = logits.shape[0]
                p = 0.0000661
                threshold = 10e-4
                for i in range(n + 1):
                    if binom.sf(i - 1, n, p) <= threshold:
                        k = i
                        break
                return bitacc >= k
            elif aggregation == 'detection-median':
                return bitacc >= logits.shape[0] // 2

    def video_embedder(
        self,
        imgs: torch.Tensor,
        msg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generates deltas one every step_size frames, then repeats for the next step_size frames.
        """
        # TODO: deal with the case where the embedder predicts images instead of deltas
        msg = msg.repeat(len(imgs) // self.step_size, 1)  # n k
        preds_w = self.embedder(imgs[::self.step_size], msg)  # n 3 h w
        preds_w = torch.repeat_interleave(
            preds_w, self.step_size, dim=0)
        return preds_w[:len(imgs)]  # f 3 h w

    def video_forward(
        self,
        imgs: torch.Tensor,  # [frames, c, h, w] for a single video
        masks: torch.Tensor,
        msgs: torch.Tensor = None,  # 1 message per video
    ) -> dict:
        """
        Generate watermarked video from the input video imgs.
        """
        # create message 1 message per video but repeat for all frames
        # we need this to calcualte the loss
        if msgs is None:
            msgs = self.get_random_msg()  # 1 x k
        else:
            assert msgs.shape[0] == 1, "Message should be unique"
        msgs = msgs.to(imgs.device)
        # generate watermarked images
        if self.embedder.yuv:  # take y channel only
            preds_w = self.video_embedder(self.rgb2yuv(imgs)[:, 0:1], msgs)
        else:
            preds_w = self.video_embedder(imgs, msgs)
        imgs_w = self.blend(imgs, preds_w)  # frames c h w
        # augment
        imgs_aug, masks, selected_aug = self.augmenter(
            imgs_w, imgs, masks, is_video=True)
        # detect watermark
        preds = self.detector(imgs_aug)
        # create and return outputs
        outputs = {
            # message per video but repeated for batchsize: b x k
            "msgs": msgs.expand(imgs.shape[0], -1),
            "masks": masks,  # augmented masks: frames 1 h w
            "imgs_w": imgs_w,  # watermarked imgs: frames c h w
            "imgs_aug": imgs_aug,  # augmented imgs: frames c h w
            "preds": preds,  # predicted message: 1 (1+nbits) h w
            "selected_aug": selected_aug,  # selected augmentation
        }
        return outputs

    @torch.no_grad()
    def embed(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor = None,
        is_video: bool = True,
    ) -> dict:
        """ 
        Generates watermarked videos from the input images and messages (used for inference).
        Videos may be arbitrarily sized.
        """
        if not is_video:
            # fallback on parent class for batch of images
            return super().embed(imgs, msgs)
        if msgs is None:
            # msgs = self.get_random_msg()  # 1 x k
            msgs = torch.zeros(1, 96)
            msgs[:, 1::2] = 1
        else:
            assert msgs.shape[0] == 1, "Message should be unique"
        msgs = msgs.repeat(self.chunk_size, 1)  # 1 k -> n k

        # encode by chunk of cksz imgs, propagate the wm to spsz next imgs
        chunk_size = self.chunk_size  # n=cksz
        step_size = self.step_size  # spsz

        # initialize watermarked imgs
        imgs_w = torch.zeros_like(imgs)  # f 3 h w

        # chunking is necessary to avoid memory issues (when too many frames)
        for ii in range(0, len(imgs[::step_size]), chunk_size):
            nimgs_in_ck = min(chunk_size, len(imgs[::step_size]) - ii)
            start = ii * step_size
            end = start + nimgs_in_ck * step_size
            all_imgs_in_ck = imgs[start: end, ...]  # f 3 h w

            # choose one frame every step_size
            imgs_in_ck = all_imgs_in_ck[::step_size]  # n 3 h w

            # deal with last chunk that may have less than chunk_size imgs
            if nimgs_in_ck < chunk_size:
                msgs = msgs[:nimgs_in_ck]

            # get deltas for the chunk, and repeat them for each frame in the chunk
            outputs = super().embed(imgs_in_ck, msgs)  # n 3 h w
            deltas_in_ck = outputs["preds_w"]  # n 3 h w
            deltas_in_ck = torch.repeat_interleave(
                deltas_in_ck, step_size, dim=0)  # f 3 h w

            # at the end of video there might be more deltas than needed
            deltas_in_ck = deltas_in_ck[:len(all_imgs_in_ck)]

            # create watermarked imgs
            all_imgs_in_ck_w = self.blend(all_imgs_in_ck, deltas_in_ck)
            imgs_w[start: end, ...] = all_imgs_in_ck_w  # n 3 h w

        outputs = {
            "imgs_w": imgs_w,  # watermarked imgs: f 3 h w
            "msgs": msgs[0:1].repeat(len(imgs), 1),  # original messages: f k
        }
        return outputs

    @torch.no_grad()
    def detect(
            self,
            imgs: torch.Tensor,
            aggregation,
            is_video: bool = True,
    ) -> dict:
        """
        Performs the forward pass of the detector only.
        Rescales the input images to 256x... pixels and then computes the mask and the message.
        Args:
            imgs (torch.Tensor): Batched images with shape FxCxHxW, where F is the number of frames,
                                    C is the number of channels, H is the height, and W is the width.
                                    if shape CxHxW is passed automatically will be considered as img
        Returns:
            dict: The output predictions.
                - torch.Tensor: Predictions for each frame with shape Fx(K+1),
                                where K is the length of the binary message. The first column represents
                                the probability of the detection bit, and the remaining columns represent
                                the probabilities of each bit in the message.
        """
        bitaccs = np.zeros(imgs.shape[0])
        for idx in range(imgs.shape[0]):
            signle_video = imgs[idx]
            if not is_video or len(signle_video.shape) == 3:
                # fallback on parent class for batch of images
                return super().detect(signle_video)
            all_preds = []
            for ii in range(0, len(signle_video), self.chunk_size):
                nimgs_in_ck = min(self.chunk_size, len(signle_video) - ii)
                outputs = super().detect(
                    signle_video[ii:ii + nimgs_in_ck]
                )
                preds = outputs["preds"]
                all_preds.append(preds)  # n k ..
            preds = torch.cat(all_preds, dim=0)  # f k ..
            outputs = {
                "preds": preds,  # predicted masks and/or messages: f (1+nbits) h w
            }

            logits = outputs["preds"][:, 1:]
            watermark_gt = np.array([0, 1] * 48)
            watermark_gt = np.tile(watermark_gt, (logits.shape[0], 1))

            # ### BA-mean
            if aggregation == 'ba-mean':
                bitacc = np.mean((watermark_gt) == (logits.detach().numpy() >= 0))

            ### BA-median
            elif aggregation == 'ba-median':
                bitacc = np.mean((logits.detach().numpy() >= 0) == watermark_gt, axis=1)
                bitacc = np.median(bitacc)

            ### Logit-mean
            elif aggregation == 'logit-mean':
                decoded = np.mean(logits.detach().numpy(), axis=0)
                bitacc = np.sum((watermark_gt) == (decoded >= 0)) / (logits.shape[0] * logits.shape[1])

            ### Logit-median
            elif aggregation == 'logit-median':
                from scipy.optimize import minimize
                def objective(median):
                    return np.sum(np.linalg.norm(logits.detach().numpy() - median, axis=1))
                initial_guess = np.mean(logits.detach().numpy(), axis=0)
                decoded = minimize(objective, initial_guess, method='Powell', tol=1e-5)
                bitacc = np.sum((watermark_gt) == (decoded.x >= 0)) / (logits.shape[0] * logits.shape[1])

            ### Bit-median
            elif aggregation == 'bit-median':
                decode = (logits.detach().numpy() >= 0)
                decoded = np.sum(decode, axis=0) > (len(decode) / 2)
                bitacc = np.sum((watermark_gt) == decoded) / (logits.shape[0] * logits.shape[1])

            ### Detection-threshold & median
            elif aggregation == 'detection-threshold' or aggregation == 'detection-median':
                bitacc = np.mean((logits.detach().numpy() >= 0) == watermark_gt, axis=1)
                bitacc = np.sum(bitacc >= 67 / 96) / logits.shape[0]

            bitaccs[idx] = bitacc

        eps = 1e-6
        p = np.clip(bitaccs, eps, 1 - eps).reshape(-1, 1)
        logit_pos = np.log(p / (1 - p))
        logit_neg = np.zeros_like(logit_pos)

        return np.concatenate([logit_neg, logit_pos], axis=1)

        # return outputs

    # @torch.no_grad()
    def detect1(
            self,
            imgs: torch.Tensor,
            is_video: bool = True,
    ) -> dict:
        """
        Performs the forward pass of the detector only.
        Rescales the input images to 256x... pixels and then computes the mask and the message.
        Args:
            imgs (torch.Tensor): Batched images with shape FxCxHxW, where F is the number of frames,
                                    C is the number of channels, H is the height, and W is the width.
                                    if shape CxHxW is passed automatically will be considered as img
        Returns:
            dict: The output predictions.
                - torch.Tensor: Predictions for each frame with shape Fx(K+1),
                                where K is the length of the binary message. The first column represents
                                the probability of the detection bit, and the remaining columns represent
                                the probabilities of each bit in the message.
        """
        if not is_video or len(imgs.shape) == 3:
            # fallback on parent class for batch of images
            return super().detect(imgs)
        all_preds = []
        for ii in range(0, len(imgs), self.chunk_size):
            nimgs_in_ck = min(self.chunk_size, len(imgs) - ii)
            outputs = super().detect(
                imgs[ii:ii + nimgs_in_ck]
            )
            preds = outputs["preds"]
            all_preds.append(preds)  # n k ..
        preds = torch.cat(all_preds, dim=0)  # f k ..
        outputs = {
            "preds": preds,  # predicted masks and/or messages: f (1+nbits) h w
        }
        return outputs

    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()

    def extract_message(
        self,
        imgs: torch.Tensor,
        aggregation: str = "avg"
    ) -> torch.Tensor:
        """
        Detects the message in a video and aggregates the predictions across frames.
        This method is mainly used for downstream inference to simplify the interface.
        If you want to obtain normal probabilities, use `video_detect` instead.
        Args:
            imgs (torch.Tensor): Batched images with shape FxCxHxW, where F is the number of frames,
                    C is the number of channels, H is the height, and W is the width.
            aggregation (str, optional): Aggregation method. Can be one of "avg",
                "squared_avg", etc. or None. Defaults to "avg".
        Returns:
            torch.Tensor: Aggregated binary message with shape k,
                where k is the length of the message.
        Note:
            If aggregation is None, returns the predictions for each frame without aggregation.
        """
        outputs = self.detect(imgs, is_video=True)
        preds = outputs["preds"]
        mask_preds = preds[:, 0:1]  # binary detection bit (not used for now)
        bit_preds = preds[:, 1:]  # f k .., must <0 for bit 0 and >0 for bit 1
        if aggregation is None:
            decoded_msg = bit_preds
        elif aggregation == "avg":
            decoded_msg = bit_preds.mean(dim=0)  # f k -> k
        msg = (decoded_msg > 0).squeeze().unsqueeze(0).to(int)  # 1 k
        return msg
