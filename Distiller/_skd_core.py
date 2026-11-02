"""Project SKD loss, preserving its reduced-loss confidence masking semantics."""
import torch
import torch.nn.functional as F
from ._added_common import kd_loss


def skd_loss(student, teacher, temperature=4., tik=0.1, sqrt_floor=1e-12):
    n, classes = student.shape
    if n < 2 or tik <= 0:
        raise ValueError('SKD covariance requires batch >= 2 and positive Tikhonov factor.')
    probabilities = F.softmax(teacher.detach(), dim=1)
    sample_confidence = probabilities.max(dim=1).values
    sample_mask = sample_confidence.le(torch.quantile(sample_confidence, .5)).float()
    class_confidence = probabilities.sum(dim=0)
    class_mask = class_confidence.le(torch.quantile(class_confidence, .5)).float()
    s, t = F.normalize(student, dim=1), F.normalize(teacher, dim=1)
    diff = (s @ s.T - t @ t.T) / classes
    centered = diff - diff.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (n - 1)
    covariance = covariance + tik * torch.eye(n, device=diff.device, dtype=diff.dtype)
    cholesky = torch.linalg.cholesky(covariance)
    # Solve instead of materializing an inverse; same quadratic form.
    solved = torch.cholesky_solve(diff.T, cholesky).T
    mahalanobis = (diff * solved).sum(dim=1).clamp_min(sqrt_floor).sqrt().mean()
    # The supplied SKD applies masks after scalar reduction. Preserve that
    # behavior explicitly rather than silently turning it into per-sample KD.
    return kd_loss(student, teacher, temperature) * sample_mask.mean() + mahalanobis * class_mask.mean()
