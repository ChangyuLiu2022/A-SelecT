#!/usr/bin/env python3

"""Image transformations."""
import torchvision as tv


def get_transforms(split, size):
    normalize = tv.transforms.Normalize(
        #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )
    if size == 448:
        resize_dim = 512
        crop_dim = 448
    elif size == 224:
        resize_dim = 256
        crop_dim = 224
    elif size == 384:
        resize_dim = 438
        crop_dim = 384
    elif size ==256:
        resize_dim = 320
        crop_dim = 256
    elif size ==512:
        resize_dim = 576
        crop_dim = 512
    elif size == 1024:
        resize_dim = 1280
        crop_dim = 1024
    if split == "train":
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),         
                tv.transforms.RandomCrop(crop_dim),       #what if centercrop?
                tv.transforms.RandomHorizontalFlip(0.5),
                # tv.transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                # tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),
                tv.transforms.CenterCrop(crop_dim),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    return transform
