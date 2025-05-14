import torch
from tqdm import tqdm
from config import *
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F


def con_loss(features, labels):
    eps = 1e-6
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    neg_label_matrix_new = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = 1 + cos_matrix
    margin = 0.3
    sim = (1 + cos_matrix) / 2.0
    scores = 1 - sim
    positive_scores = torch.where(pos_label_matrix == 1.0, scores, scores - scores)
    mask = torch.eye(features.size(0)).cuda()
    positive_scores = torch.where(mask == 1.0, positive_scores - positive_scores, positive_scores)
    positive_scores = torch.sum(positive_scores, dim=1, keepdim=True) / (
                (torch.sum(pos_label_matrix, dim=1, keepdim=True) - 1) + eps)
    positive_scores = torch.repeat_interleave(positive_scores, B, dim=1)
    relative_dis1 = margin + positive_scores - scores
    neg_label_matrix_new[relative_dis1 < 0] = 0
    neg_label_matrix = neg_label_matrix * neg_label_matrix_new
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= B * B

    return loss


# def visualize_crop(image_tensor, coords, save_path):
#     image_np = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
#     image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#     x_min, y_min, x_max, y_max = map(int, coords)
#     h, w = image_bgr.shape[:2]
#     x_min = max(0, min(x_min, w-1))
#     y_min = max(0, min(y_min, h-1))
#     x_max = max(x_min+1, min(x_max, w))
#     y_max = max(y_min+1, min(y_max, h))
#     cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
#     cv2.imwrite(save_path, image_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, criterion_mix, alpha, epoch):
    model.train()
    total_loss = 0.0
    for step, data in enumerate(tqdm(train_loader, desc='Training %d epoch' % epoch)):
        images, labels = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()

        # Stage 1
        out110, out111, out112, out113, fore_x, back_x, _, image_new, lam_a, lam_b, rand_index = model(images, swap=True)

        # if step == 0:
        #     for i in range(min(4, images.size(0))):
        #         img = images[i].cpu().detach()
        #         coords = crop_coords[i]
        #         visualize_crop(img, coords, f"outputs/cub/epoch{epoch}_step{step}_sample{i}.jpg")

        # Stage 2
        out210, out211, out212, out213 = model(fore_x.detach(), swap=False)
        _, _, _, out313 = model(back_x.detach(), swap=False)

        # Stage 3
        out410, out411, out412, out413 = model(image_new.detach(), swap=False)
        target_b = labels[rand_index]

        fore_back_loss = con_loss(torch.cat((out113 + out213, out313), 0), torch.cat((labels, torch.zeros([batch_size], dtype = torch.long).fill_(-1).cuda()), 0))

        loss_stage1 = (criterion(out110, labels) * alpha + criterion(out111, labels) * alpha +
                       criterion(out112, labels) + criterion(out113, labels))
        loss_stage2 = (criterion(out210, labels) * alpha + criterion(out211, labels) * alpha +
                       criterion(out212, labels) + criterion(out213, labels))
        loss_stage3 = torch.mean(criterion_mix(out410, labels) * lam_a + criterion_mix(out410, target_b) * lam_b) * alpha + \
                    torch.mean(criterion_mix(out411, labels) * lam_a + criterion_mix(out411, target_b) * lam_b) * alpha + \
                    torch.mean(criterion_mix(out412, labels) * lam_a + criterion_mix(out412, target_b) * lam_b) + \
                    torch.mean(criterion_mix(out413, labels) * lam_a + criterion_mix(out413, target_b) * lam_b)
        loss = loss_stage1 + loss_stage2 + loss_stage3 + fore_back_loss
        print("loss_stage1:{:.4f}, loss_stage2:{:.4f}, loss_stage3:{:.4f}, fore_back_loss:{:.4f}".format(loss_stage1.item(), loss_stage2.item(), loss_stage3.item(), fore_back_loss.item()))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}')


def valid(model, test_loader, epoch, beta=None):
    if beta is None:
        beta = [1, 1, 1, 1]
    test_total = len(test_loader.dataset)
    acc = [0] * 5
    with torch.no_grad():
        model.eval()
        for step, data in enumerate(tqdm(test_loader, desc='Test %d epoch' % epoch)):
            images, labels = data[0].cuda(), data[1].cuda()
            outs = model(images, swap=False)
            for i in range(4):
                prediction = outs[i].argmax(dim=1)
                acc[i] += torch.eq(prediction, labels).sum().float().item()
            weighted_prediction = (outs[0] * beta[0] + outs[1] * beta[1] +
                                   outs[2] * beta[2] + outs[3] * beta[3]).argmax(dim=1)
            acc[4] += torch.eq(weighted_prediction, labels).sum().float().item()
        acc = [a / test_total for a in acc]
        info = '[epoch {}] ACC: {:.4f} {:.4f} {:.4f} {:.4f} || {:.4f}'.format(epoch, acc[0], acc[1], acc[2], acc[3],
                                                                              acc[4])
        print(info)
