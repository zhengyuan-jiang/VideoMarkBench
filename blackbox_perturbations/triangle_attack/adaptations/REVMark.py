import torch
import torch.nn as nn
from TAsBlock import TAsBlock

import numpy as np

class Encoder(nn.Module):
    def __init__(self, msgbitnum, videoshape):
        super().__init__()
        self.videoshape = videoshape
        self.msg_reshape = nn.Sequential(
            nn.Linear(msgbitnum, 3*4*(videoshape[1]//8)*(videoshape[2]//8)), nn.LeakyReLU(inplace=True)
        )
        self.msg_up = nn.Upsample(scale_factor=(2,8,8))
        self.tasblock = TAsBlock(framenum=videoshape[0], enable=False)
        self.up = nn.Upsample(scale_factor=(2,2,2), mode='nearest')
        self.conv1 = nn.Sequential(nn.Conv3d(9, 16, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.LeakyReLU(inplace=True), nn.Conv3d(32, 32, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.LeakyReLU(inplace=True), nn.Conv3d(64, 64, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.LeakyReLU(inplace=True), nn.Conv3d(128, 128, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv3d(128+64, 64, 3, padding=1), nn.LeakyReLU(inplace=True), nn.Conv3d(64, 64, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv3d(64+32, 32, 3, padding=1), nn.LeakyReLU(inplace=True), nn.Conv3d(32, 32, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv3d(32+16, 16, 3, padding=1), nn.LeakyReLU(inplace=True), nn.Conv3d(16, 16, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True))
        self.conv8 = nn.Sequential(nn.Conv3d(16+3+3, 16, 3, padding=1), nn.LeakyReLU(inplace=True), nn.Conv3d(16, 3, 1))
    
    def forward(self, video, msgbit): # video:(B,C,F,H,W)
        msg = self.msg_reshape(msgbit - 0.5).reshape(-1, 3, 4, self.videoshape[1]//8, self.videoshape[2]//8)
        msg = self.msg_up(msg)
        feat = self.tasblock(video)
        conv1 = self.conv1(torch.cat([msg, video, feat], dim=1))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(torch.cat([conv3, self.up(conv4)], dim=1))
        conv6 = self.conv6(torch.cat([conv2, self.up(conv5)], dim=1))
        conv7 = self.conv7(torch.cat([conv1, self.up(conv6)], dim=1))
        return self.conv8(torch.cat([conv7, msg, video], dim=1))

class Decoder(nn.Module):
    def __init__(self, msgbitnum, videoshape):
        super().__init__()
        self.tasblock = TAsBlock(framenum=videoshape[0], enable=False)
        self.decoder1 = nn.Sequential(
            nn.Conv3d(6, 16, 3, stride=(1,2,2), padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 32, 3, stride=(1,2,2), padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=(2,2,2), padding=1), nn.LeakyReLU(inplace=True), nn.Conv3d(64, 64, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 128, 3, stride=(2,2,2), padding=1), nn.LeakyReLU(inplace=True), nn.Conv3d(128, 128, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 256, 3, stride=(2,2,2), padding=1), nn.LeakyReLU(inplace=True), nn.Conv3d(256, 256, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(256*(videoshape[1]//32)*(videoshape[2]//32), 512), nn.LeakyReLU(inplace=True),
            nn.Linear(512, msgbitnum),
            nn.Sigmoid()
        )

    def forward(self, video, aggregation):
        B, FKC, H, W = video.shape
        video = video.reshape(FKC//24, 3, 8, H, W)
        feat = self.tasblock(video)
        d = self.decoder1(torch.cat([video, feat], dim=1))
        logits = self.decoder2(d.reshape(d.shape[0], -1))

        watermark_gt = np.array([1, 0] * 48)
        watermark_gt = np.tile(watermark_gt, (logits.shape[0], 1))
        bitacc = np.mean((watermark_gt) == (logits.detach().cpu().numpy() >= 0.5))
        return bitacc >= 67/96

    def detect(self, imgs):
        bitaccs = np.zeros(imgs.shape[0])
        for idx in range(imgs.shape[0]):
            image = imgs[idx]

            feat = self.tasblock(image)
            d = self.decoder1(torch.cat([image, feat], dim=1))
            logits = self.decoder2(d.reshape(d.shape[0], -1))

            watermark_gt = np.array([1, 0] * 48)
            watermark_gt = np.tile(watermark_gt, (logits.shape[0], 1))
            bitacc = np.mean((watermark_gt) == (logits.detach().cpu().numpy() >= 0.5))  # Attack BA-mean aggregation
            bitaccs[idx] = bitacc

        eps = 1e-6
        p = np.clip(bitaccs, eps, 1 - eps).reshape(-1, 1)
        logit_pos = np.log(p / (1 - p))
        logit_neg = np.zeros_like(logit_pos)

        return np.concatenate([logit_neg, logit_pos], axis=1)

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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d( 3, 16, 3, padding=1), nn.LeakyReLU(inplace=True), nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(16, 32, 3, padding=1), nn.LeakyReLU(inplace=True), nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(32, 64, 3, padding=1), nn.LeakyReLU(inplace=True), nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(64,128, 3, padding=1), nn.LeakyReLU(inplace=True), nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(128, 1, 1)
        )
    def forward(self, image):
        score = self.model(image).mean()
        return score

def framenorm(batch): # batch:(B,C,F,H,W)
    m, s = torch.mean(batch), torch.std(batch)
    batch = batch.clamp(m-s*7, m+s*7)
    m, s = torch.mean(batch), torch.std(batch)
    batch = batch.clamp(m-s*7, m+s*7)
    return batch / (torch.sqrt(torch.sum(batch**2, dim=[1,3,4], keepdim=True)) + 1e-3*s)


if __name__ == '__main__':
    device = 'cuda'
    encoder = Encoder(96, [8,128,128]).to(device)
    decoder = Decoder(96, [8,128,128]).to(device)
    x = torch.rand([1,3,8,128,128]).to(device)*2-1
    m = torch.randint(0, 2, (1,96)).float().to(device)
    r = encoder(x, m)
    y = x + 6.2*framenorm(r)
    d = decoder(y)
    