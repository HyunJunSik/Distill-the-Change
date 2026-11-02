"""InDistill classification port: ordered layer warmup and fixed L1 targets.

Sources: gsarridis/InDistill, commit 70f47f063756f52e0bb33c3803350c597d258188;
WACV 2025 paper (arXiv 2205.10003v4), equations 1-4. See SOURCES.md for
differences from the upstream CIFAR-10/PKT example and pruning semantics.
"""
import copy

import torch
from torch import nn
import torch.nn.functional as F

from ._added_features import stages


def widen_student(student):
    """Same computation topology as student, 2x channel widths; classifier stays 100-way."""
    model = copy.deepcopy(student)
    for path, module in list(model.named_modules()):
        if not path:
            continue
        replacement = None
        if isinstance(module, nn.Conv2d):
            cin = 3 if module.in_channels == 3 else 2 * module.in_channels
            cout = 2 * module.out_channels
            groups = 2 * module.groups if module.groups == module.in_channels and module.in_channels != 3 else module.groups
            replacement = nn.Conv2d(cin, cout, module.kernel_size, module.stride, module.padding,
                                    module.dilation, groups, module.bias is not None, module.padding_mode)
        elif isinstance(module, nn.BatchNorm2d):
            replacement = nn.BatchNorm2d(2 * module.num_features, eps=module.eps, momentum=module.momentum,
                                         affine=module.affine, track_running_stats=module.track_running_stats)
        elif isinstance(module, nn.Linear):
            if module.out_features != 100:
                raise ValueError('This auxiliary builder supports the current single 100-way classifier only.')
            replacement = nn.Linear(2 * module.in_features, 100, bias=module.bias is not None)
        if replacement is not None:
            parent, _, leaf = path.rpartition('.')
            target = model.get_submodule(parent) if parent else model
            target._modules[leaf] = replacement
    for module in model.modules():
        if hasattr(module, 'nChannels'):
            module.nChannels *= 2
        if hasattr(module, 'stage_channels'):
            old = module.stage_channels
            module.stage_channels = type(old)(2 * x for x in old)
    return model


def conv_score(conv):
    return conv.weight.detach().float().abs().flatten(1).sum(dim=1)


def shuffle_score(score):
    return score.reshape(2, -1).T.reshape(-1)


def stage_filter_scores(model, name):
    if name == 'mobilenetv1':
        return [conv_score(model.model[i][3]) for i in (3, 5, 11)]
    if name == 'ShuffleV2':
        scores = []
        for stage in (model.layer1, model.layer2, model.layer3):
            first = stage[0]
            score = shuffle_score(torch.cat((conv_score(first.conv2), conv_score(first.conv5))))
            for block in list(stage)[1:]:
                half = score.numel() // 2
                score = shuffle_score(torch.cat((score[:half], conv_score(block.conv3))))
            scores.append(score)
        return scores
    if name == 'ShuffleV1':
        # Each stage ends with a stride-1 residual block whose final grouped
        # pointwise convolution has the full stage channel count.
        return [conv_score(stage[-1].conv3) for stage in (model.layer1, model.layer2, model.layer3)]
    if name.startswith('vgg'):
        return [conv_score(next(m for m in reversed(list(getattr(model, f'block{i}').modules()))
                                if isinstance(m, nn.Conv2d))) for i in (1, 2, 3, 4)]
    if name.startswith('wrn'):
        return [conv_score(block.layer[-1].conv2) for block in (model.block1, model.block2, model.block3)]
    if name.startswith('resnet'):
        return [conv_score(layer[-1].conv2) for layer in (model.layer1, model.layer2, model.layer3)]
    raise ValueError(f'No fixed L1 stage ranking for {name}.')


class InformationTargets(nn.Module):
    def __init__(self, teacher, teacher_name, student_name, fs, ft, a=5, b=1):
        super().__init__()
        self.student_name, self.teacher_name = student_name, teacher_name
        ss, ts = stages(student_name, fs), stages(teacher_name, ft)
        if len(ss) != len(ts):
            raise ValueError('InDistill requires aligned stages; an auxiliary teacher is needed for this pair.')
        self.durations = [a + (i + 1) * b for i in range(len(ss))]
        if any(x <= 0 for x in self.durations):
            raise ValueError('Positive InDistill stage durations required.')
        self.warmup_epochs = sum(self.durations)
        scores = stage_filter_scores(teacher, teacher_name) if any(s.shape[1] != t.shape[1] for s, t in zip(ss, ts)) else None
        layout = []
        for i, (s, t) in enumerate(zip(ss, ts)):
            if s.shape[-2:] != t.shape[-2:] or s.shape[1] > t.shape[1]:
                raise ValueError('InDistill targets must match spatial sizes and be at least as wide as the student.')
            width = s.shape[1]
            if width == t.shape[1]:
                keep = torch.arange(width)
            else:
                score = scores[i].cpu()
                if len(score) != t.shape[1]:
                    raise ValueError('L1 filter score does not match feature channels.')
                keep = torch.argsort(score, descending=True, stable=True)[:width].sort().values
            self.register_buffer(f'keep_{i}', keep)
            layout.append(dict(student_shape=list(s.shape[1:]), teacher_shape=list(t.shape[1:]), kept_channels=keep.tolist()))
        self.layout = dict(stages=layout, stage_epochs=self.durations, warmup_epochs=self.warmup_epochs,
                           target_pruning='fixed L1-selected stage output channels; full teacher path retained',
                           feature_alignment='no trainable projection; no spatial resizing')

    def stage_for_epoch(self, epoch):
        end = 0
        for i, duration in enumerate(self.durations):
            end += duration
            if epoch < end:
                return i
        return None

    def forward(self, fs, ft, n, epoch):
        index = self.stage_for_epoch(epoch)
        if index is None:
            raise ValueError('Intermediate MSE requested after InDistill warmup.')
        s = stages(self.student_name, fs)[index][:n].float()
        t = stages(self.teacher_name, ft)[index][:n].float().index_select(1, getattr(self, f'keep_{index}'))
        return F.mse_loss(s, t)


def set_trainable_prefix(student, name, index):
    """During stage i train first i+1 stages; restore full model after warmup."""
    if index is None:
        student.requires_grad_(True)
        student.train()
        return
    student.requires_grad_(False)
    student.eval()
    if name == 'mobilenetv1':
        active = list(student.model)[:(4, 6, 12)[index]]
    elif name.startswith('vgg'):
        # Stage index 0 corresponds to block1; block0 is the stem.
        active = [getattr(student, f'block{i}') for i in range(index + 2)]
        active += [getattr(student, f'pool{i}') for i in range(index + 1)]
    elif name.startswith('wrn'):
        active = [student.conv1] + [student.block1, student.block2, student.block3][:index + 1]
    else:
        active = [student.conv1, student.bn1] + [student.layer1, student.layer2, student.layer3][:index + 1]
    for module in active:
        module.requires_grad_(True)
        module.train()
