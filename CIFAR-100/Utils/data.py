"""Paired data order and augmentations, independent of method/worker count."""
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from augmentation import RandAugment

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


@contextmanager
def data_rng(seed):
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class PairedTrainingData(Dataset):
    def __init__(self, raw, indices, seed, base_augmentation="concat"):
        self.raw, self.indices, self.seed = raw, list(indices), seed
        self.epoch = 0
        self.concat = base_augmentation == "concat"
        common = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        tail = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.weak = transforms.Compose(common + tail)
        self.strong = transforms.Compose(common + [RandAugment(2, 10)] + tail)

    def __len__(self):
        return len(self.indices) * (2 if self.concat else 1)

    def __getitem__(self, index):
        image, label = self.raw[self.indices[index % len(self.indices)]]
        transform = self.strong if index >= len(self.indices) else self.weak
        # The concatenated view index is included so weak/strong draws differ.
        seed = (self.seed * 1000003 + self.epoch * 2000003 + index * 97) % (2**63 - 1)
        with data_rng(seed):
            return transform(image), label


class EvaluationData(Dataset):
    def __init__(self, raw, indices):
        self.raw, self.indices = raw, list(indices)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.raw[self.indices[index]]
        return self.transform(image), label


def stratified_split(targets, val_per_class, seed):
    labels = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if not 0 < val_per_class < len(indices):
            raise ValueError("val_per_class must leave at least one training sample per class")
        val.extend(indices[:val_per_class].tolist())
        train.extend(indices[val_per_class:].tolist())
    return sorted(train), sorted(val)


def make_data(args, seed):
    if args.smoke:
        raw = datasets.FakeData(size=40, image_size=(3, 32, 32), num_classes=100, random_offset=77)
        train_indices, val_indices = list(range(24)), list(range(24, 32))
        raw_test = datasets.FakeData(size=8, image_size=(3, 32, 32), num_classes=100, random_offset=999)
    else:
        raw = datasets.CIFAR100(root=str(args.data_root), train=True, download=False)
        raw_test = datasets.CIFAR100(root=str(args.data_root), train=False, download=False)
        train_indices, val_indices = stratified_split(raw.targets, args.val_per_class, args.split_seed)
    train = PairedTrainingData(raw, train_indices, seed, args.base_augmentation)
    val = EvaluationData(raw, val_indices)
    test = EvaluationData(raw_test, range(len(raw_test)))
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=args.device.startswith("cuda"), persistent_workers=False)
    # Never drop samples; the default 90,000 examples do not end in a singleton.
    if len(train) % args.batch_size == 1:
        raise ValueError("Final training batch has one sample; choose another batch size for relational controls.")
    train_loader = DataLoader(train, shuffle=True, generator=torch.Generator(), **kwargs)
    val_loader = DataLoader(val, shuffle=False, generator=torch.Generator().manual_seed(0), **kwargs)
    test_loader = DataLoader(test, shuffle=False, generator=torch.Generator().manual_seed(0), **kwargs)
    split = {"train": train_indices, "validation": val_indices, "split_seed": args.split_seed}
    return train_loader, val_loader, test_loader, split


def prepare_epoch(loader, seed, epoch):
    loader.dataset.epoch = epoch
    loader.generator.manual_seed(seed * 1000003 + epoch * 2000003)

