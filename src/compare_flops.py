import os
import argparse
from functools import lru_cache

import torch
from fvcore.nn import FlopCountAnalysis

from models.vgg import cifar10_vgg16_bn as st_vgg16_model
from models.resnet import ResNet18 as st_ResNet18_model
from models.mobilenet import MobileNet as st_MobileNet_model
from models_cnnsplitter.simcnn import SimCNN as st_simcnn_model
from models_cnnsplitter.rescnn import ResCNN as st_rescnn_model

from modules_arch.vgg_module_v2 import cifar10_vgg16_bn as vgg16_module
from modules_arch.resnet_module_v2 import ResNet18 as ResNet18_module
from modules_arch.mobilenet_module_v2 import mobilenet_module as MobileNet_module
from modules_arch.simcnn_module import SimCNN as simcnn_module
from modules_arch.rescnn_module import ResCNN as rescnn_module

from configs import Configs
from modularizer import get_args

DEVICE = torch.device('cuda')


def generate_target_module(num_classes, target_classes, module_mask_path):
    global mt_model_save_path
    all_masks = torch.load(module_mask_path, map_location=DEVICE)
    cls_mask = (all_masks[target_classes].sum(dim=0) > 0).int()
    mt_model_param = torch.load(mt_model_save_path, map_location=DEVICE)

    if model_name == 'vgg16':
        m = vgg16_module(model_param=mt_model_param, module_mask=cls_mask, keep_generator=False, num_classes=num_classes)
    elif model_name == 'resnet18':
        m = ResNet18_module(model_param=mt_model_param, module_mask=cls_mask, keep_generator=False, num_classes=num_classes)
    elif model_name == 'mobilenet':
        m = MobileNet_module(model_param=mt_model_param, module_mask=cls_mask, keep_generator=False, num_classes=num_classes)
    elif model_name == 'simcnn':
        m = simcnn_module(model_param=mt_model_param, module_mask=cls_mask, keep_generator=False, num_classes=num_classes)
    elif model_name == 'rescnn':
        m = rescnn_module(model_param=mt_model_param, module_mask=cls_mask, keep_generator=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if hasattr(m, 'module_head'):
        m.module_head = torch.nn.Identity()
    return m.to(DEVICE)


@lru_cache()
def load_standard_model(num_classes):
    if model_name == 'vgg16':
        return st_vgg16_model(pretrained=False, num_classes=num_classes).to(DEVICE)
    elif model_name == 'resnet18':
        return st_ResNet18_model(num_classes=num_classes).to(DEVICE)
    elif model_name == 'mobilenet':
        return st_MobileNet_model(num_classes=num_classes).to(DEVICE)
    elif model_name == 'simcnn':
        return st_simcnn_model(num_classes=num_classes).to(DEVICE)
    elif model_name == 'rescnn':
        return st_rescnn_model(num_classes=num_classes).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def calculate_flops(num_classes, target_classes, masks_path):
    std_model = load_standard_model(num_classes)
    std_model.eval()
    composed_model = generate_target_module(num_classes, target_classes, masks_path)
    composed_model.eval()

    # x = torch.randn(1, 3, 32, 32, device=DEVICE)
    x = torch.randn(1, 3, 224, 224, device=DEVICE)
    std_total_flops = FlopCountAnalysis(std_model, x).unsupported_ops_warnings(False).total()
    com_total_flops = FlopCountAnalysis(composed_model, x).unsupported_ops_warnings(False).total()

    ratio = com_total_flops / std_total_flops if std_total_flops > 0 else float('nan')
    print(
        f"COM_MODEL_FLOPS/STD_MODEL_FLOPS: "
        f"{com_total_flops}/{std_total_flops}="
        f"{ratio:.4f}  -------- {target_classes}"
    )

def generate_model_composition_tasks(num_classes):
    file_name = f"target_classes.num_classes_{num_classes}.rep_tasks.list"
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")
    with open(file_name, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                yield [int(x) for x in parts]

def main():
    args = get_args()
    global model_name, dataset_name, mt_model_save_path

    model_name = args.model
    dataset_name = args.dataset
    batch_size = args.batch_size
    THRESHOLD = args.threshold
    lr_m = args.lr_model
    alpha, beta = args.alpha, args.beta
    target_classes = args.target_classes

    if dataset_name in ('cifar10', 'svhn'):
        N = 10
    elif dataset_name in ('cifar100', 'imagenet'):
        N = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    cfg = Configs()
    base = os.path.join(cfg.data_dir, f"{model_name}_{dataset_name}")

    mt_model_save_path = os.path.join(
        base,
        f"lr_model_mask_{lr_m}_{lr_m}_a_{alpha}_b_{beta}_bz_{batch_size}.pth"
    )
    if not os.path.exists(mt_model_save_path):
        raise FileNotFoundError(f"Cannot find masked model file: {mt_model_save_path}")

    masks_path = os.path.join(
        base, 'modules',
        f"lr_model_mask_{lr_m}_{lr_m}_a_{alpha}_b_{beta}_bz_{batch_size}",
        f"mask_thres_{THRESHOLD}.pth"
    )
    if not os.path.exists(masks_path):
        raise FileNotFoundError(f"Cannot find mask file: {masks_path}")

    for tgt in generate_model_composition_tasks(N):
        calculate_flops(N, tgt, masks_path)

    # calculate_flops(N, target_classes, masks_path)


if __name__ == '__main__':
    main()
