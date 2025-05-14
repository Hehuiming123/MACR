import torch
import numpy as np
import torch.nn as nn
from config import *
from datasets import read_data_set
from torchvision import transforms
from PIL import Image
from ptflops import get_model_complexity_info
from train import train_one_epoch, valid
from model import VisionTransformer

import os
import random
import logging

logger = logging.getLogger(__name__)

def seed_torch(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

train_transform = transforms.Compose([
    transforms.Resize(zoom_size, Image.BILINEAR),
    transforms.RandomCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transform = transforms.Compose([
    transforms.Resize(zoom_size, Image.BILINEAR),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
train_loader = read_data_set(train_csv, train_root, batch_size, train_transform)
test_loader = read_data_set(test_csv, test_root, batch_size, test_transform)

criterion = nn.CrossEntropyLoss()
criterion_mix = nn.CrossEntropyLoss(reduction='none')

model = VisionTransformer(config, input_size, num_classes=config.num_classes)
model.load_from(np.load(vit_pretrain))
model = nn.DataParallel(model)
model = model.cuda()
print(model)

for name, param in model.named_parameters():
    print(f'Parameter name: {name}, requires_grad: {param.requires_grad}')
model.eval()
with torch.no_grad():
    macs, params = get_model_complexity_info(model, 
                                            (3, 448, 448), 
                                            as_strings=False,
                                            print_per_layer_stat=False,
                                            verbose=False)
flops = 2 * macs

print("== Final Model Complexity ==")
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Parameters: {params / 1e6:.2f} M")

fc10 = list(map(id, model.module.part_head10.parameters()))
fc11 = list(map(id, model.module.part_head11.parameters()))
norm10 = list(map(id, model.module.transformer.encoder.norm10.parameters()))
norm11 = list(map(id, model.module.transformer.encoder.norm11.parameters()))
base_params = filter(lambda p: id(p) not in fc10 + fc11 + norm10 + norm11, model.parameters())


optimizer = torch.optim.SGD([{"params": base_params, "lr": lr},
                                {"params": model.module.part_head10.parameters(), "lr": lr * lr_ml},
                                {"params": model.module.part_head11.parameters(), "lr": lr * lr_ml},
                                {"params": model.module.transformer.encoder.norm10.parameters(), "lr": lr * lr_ml},
                                {"params": model.module.transformer.encoder.norm11.parameters(), "lr": lr * lr_ml}],
                            lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)

for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer, scheduler, criterion, criterion_mix, alpha, epoch)
    torch.save(model.state_dict(), 'checkpoints/cub'.format(which_set, epoch))
    valid(model, test_loader, epoch, beta)
