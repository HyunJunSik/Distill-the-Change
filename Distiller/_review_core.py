"""Port of the project's ReviewKD: dynamic ABF shapes and HCL targets."""
import torch
from torch import nn
import torch.nn.functional as F

from ._added_features import stages


def deterministic_adaptive_avg_pool2d(x, output_size):
    """Adaptive-average bins using reshape/means, without CUDA adaptive backward.

    Bins follow floor(i*input/output):ceil((i+1)*input/output).
    Divisible sizes use a vectorized non-overlapping reduction. Floating-point
    summation order can differ from the original kernel.
    """
    oh, ow = (output_size, output_size) if isinstance(output_size, int) else output_size
    h, w = x.shape[-2:]
    if oh < 1 or ow < 1:
        raise ValueError('Positive pooling output sizes required.')
    if h % oh == 0 and w % ow == 0:
        return x.reshape(*x.shape[:-2], oh, h // oh, ow, w // ow).mean(dim=(-3, -1))
    rows = []
    for i in range(oh):
        top, bottom = i * h // oh, ((i + 1) * h + oh - 1) // oh
        cells = []
        for j in range(ow):
            left, right = j * w // ow, ((j + 1) * w + ow - 1) // ow
            cells.append(x[..., top:bottom, left:right].mean(dim=(-2, -1)))
        rows.append(torch.stack(cells, dim=-1))
    return torch.stack(rows, dim=-2)


def hcl_loss(student, teacher):
    total = student[0].new_zeros(())
    for fs, ft in zip(student, teacher):
        if fs.shape != ft.shape:
            raise ValueError(f'ReviewKD target mismatch: {fs.shape} != {ft.shape}')
        loss, coefficient, denominator = F.mse_loss(fs, ft), 1., 1.
        for size in (4, 2, 1):
            if size >= min(fs.shape[-2:]):
                continue
            coefficient /= 2.
            loss = loss + coefficient * F.mse_loss(deterministic_adaptive_avg_pool2d(fs, size), deterministic_adaptive_avg_pool2d(ft, size))
            denominator += coefficient
        total = total + loss / denominator
    return total


class ABF(nn.Module):
    def __init__(self, cin, mid, cout, fuse):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid))
        self.conv2 = nn.Sequential(nn.Conv2d(mid, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.att_conv = nn.Sequential(nn.Conv2d(2 * mid, 2, 1), nn.Sigmoid()) if fuse else None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(self, feature, residual, target_size):
        x = self.conv1(feature)
        if self.att_conv is not None:
            residual = F.interpolate(residual, size=x.shape[-2:], mode='nearest')
            attention = self.att_conv(torch.cat((x, residual), dim=1))
            x = x * attention[:, :1] + residual * attention[:, 1:]
        x = F.interpolate(x, size=target_size, mode='nearest')
        return self.conv2(x), x


class ReviewLoss(nn.Module):
    def __init__(self, student_name, teacher_name, fs, ft, max_mid=512):
        super().__init__()
        self.student_name, self.teacher_name = student_name, teacher_name
        ss, ts = stages(student_name, fs), stages(teacher_name, ft, preact=True)
        count = min(len(ss), len(ts))
        if count < 1:
            raise ValueError('ReviewKD needs convolutional feature stages.')
        # Preserve all stages when counts agree. Otherwise match depth quantiles.
        def indices(length):
            return [length - 1] if count == 1 else [round(i * (length - 1) / (count - 1)) for i in range(count)]
        self.student_indices, self.teacher_indices = indices(len(ss)), indices(len(ts))
        ss = [ss[i] for i in self.student_indices] + [fs['pooled_feat'][:, :, None, None]]
        ts = [ts[i] for i in self.teacher_indices] + [ft['pooled_feat'][:, :, None, None]]
        mid = min(max_mid, ss[-1].shape[1])
        self.abfs = nn.ModuleList([ABF(s.shape[1], mid, t.shape[1], i < len(ss) - 1)
                                  for i, (s, t) in enumerate(zip(ss, ts))][::-1])
        self.layout = dict(student_stage_indices=self.student_indices, teacher_stage_indices=self.teacher_indices,
                           student_shapes=[list(x.shape[1:]) for x in ss], teacher_shapes=[list(x.shape[1:]) for x in ts],
                           matching='equal stages or normalized-depth quantiles, plus pooled feature')

    def forward(self, fs, ft, n):
        ss = stages(self.student_name, fs)
        ts = stages(self.teacher_name, ft, preact=True)
        ss = [ss[i][:n].float() for i in self.student_indices] + [fs['pooled_feat'][:n, :, None, None].float()]
        ts = [ts[i][:n].float() for i in self.teacher_indices] + [ft['pooled_feat'][:n, :, None, None].float()]
        results, residual = [], None
        for head, s, t in zip(self.abfs, ss[::-1], ts[::-1]):
            output, residual = head(s, residual, t.shape[-2:])
            results.insert(0, output)
        return hcl_loss(results, ts)
