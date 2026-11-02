"""ImageNet class-directory input: train/<class> and val/<class>."""
import random
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset


class TrainingImages(Dataset):
    def __init__(self, root, seed):
        self.raw = ImageFolder(root)
        self.seed, self.epoch = seed, 0
        self.transform = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                                   T.Normalize((.485, .456, .406), (.229, .224, .225))])

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        seed = self.seed * 1000003 + self.epoch * 2000003 + index * 97
        try:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.default_generator.manual_seed(seed)
            image, target = self.raw[index]
            return self.transform(image), target
        finally:
            random.setstate(py)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu)


def make_data(args, seed):
    train = TrainingImages(str(args.data_root / "train"), seed)
    eval_root = args.data_root / "val"
    if not eval_root.is_dir():
        eval_root = args.data_root / "valid"
    validation = ImageFolder(str(eval_root), transform=T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize((.485, .456, .406), (.229, .224, .225))]))
    if train.raw.class_to_idx != validation.class_to_idx:
        raise ValueError("Train and validation class-directory mappings differ")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              generator=torch.Generator(), pin_memory=args.device.startswith("cuda"))
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                   generator=torch.Generator().manual_seed(0), pin_memory=args.device.startswith("cuda"))
    # ImageNet validation is reported only at the fixed final epoch; no selection.
    return train_loader, None, validation_loader, dict(train=len(train), validation=len(validation),
                                                       class_to_idx=train.raw.class_to_idx)


def prepare_epoch(loader, seed, epoch):
    loader.dataset.epoch = epoch
    loader.generator.manual_seed(seed * 1000003 + epoch * 2000003)
