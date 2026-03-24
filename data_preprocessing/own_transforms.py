
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch
class ZScaleNormalize(object):
    def __call__(self, tensor):
        return (tensor - tensor.mean()) / tensor.std()

data_stats = {
    'FMNIST': ((0.2860,), (0.3530,)),
    'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    'MCND': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'Kvasir_v2': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "Clothing1M": ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
}
def get_stransform(dataset, 
                   train,
                   normalize=True):

    color_jitter = transforms.ColorJitter(
        0.8 , 0.8, 0.8, 0.2
    )
    if train:
        if dataset in ['FMNIST']:
            transform = transforms.Compose([
                transforms.Resize(32),
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1])])
        elif dataset in ['SVHN']:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1])])
        elif dataset == 'Clothing1M':
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1]),
            ])
        elif dataset in ['FLAIR']:
            transform = transforms.Compose([
                transforms.Resize(128),
                transforms.RandomCrop(128, padding=10),
                transforms.RandomHorizontalFlip(p=0.9),
                transforms.ToTensor()])
        elif dataset == 'MCND':
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1])])
        elif dataset == 'Kvasir_v2':
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1])])
        elif dataset == 'RetinalOCT':
            transform = transforms.Compose([
                v2.Grayscale(num_output_channels=1),  # PIL -> grayscale
                v2.Resize((160, 320), antialias=True),  # PIL -> resize (faster for small images)
                v2.ToImage(),  # PIL -> tensor (uint8)
                v2.ToDtype(torch.float32, scale=True),  # tensor -> float32
                ZScaleNormalize(),
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1])])
    else:
        if dataset in ['FLAIR']:
            transform = transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor()])
        elif dataset in ['Clothing1M']:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1]),
            ])
        elif dataset == 'MCND':
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1])])
        elif dataset == 'Kvasir_v2':
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1])])
        elif dataset == 'RetinalOCT':
            transform = transforms.Compose([
                v2.Grayscale(num_output_channels=1),  # PIL -> grayscale
                v2.Resize((160, 320), antialias=True),  # PIL -> resize (faster for small images)
                v2.ToImage(),  # PIL -> tensor (uint8)
                v2.ToDtype(torch.float32, scale=True),  # tensor -> float32
                ZScaleNormalize(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1])])
    return transform

class MultiCropTransform:
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4),
                 global_crops_number=2, local_crops_number=2):  # 减少 local crop 数量
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number

        # 全局增强 (32x32 for CIFAR10)
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=global_crops_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(9, sigma=(0.1, 2.0)),  # kernel 从 23 改为 9，更快
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
        ])

        # 局部增强 (16x16 for CIFAR10)
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(16, scale=local_crops_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(5, sigma=(0.1, 2.0)),  # kernel 再缩小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
        ])

    def __call__(self, x):
        crops = [self.global_transform(x) for _ in range(self.global_crops_number)]
        crops += [self.local_transform(x) for _ in range(self.local_crops_number)]
        return crops