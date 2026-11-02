"""Stable port of the project's SKD with the legacy training return format."""
import torch
import torch.nn.functional as F

from .distiller import Distiller
from ._skd_core import skd_loss


class SKD(Distiller):
    def __init__(self, student, teacher, temp=4., tik=.1, ce=.1, kd=.9, *, sqrt_floor=1e-12):
        super().__init__(student, teacher)
        if min(temp, tik, sqrt_floor) <= 0:
            raise ValueError('temperature, tik, sqrt_floor must be positive.')
        self.temperature, self.tik, self.sqrt_floor = temp, tik, sqrt_floor
        self.ce_loss_weight, self.kd_loss_weight = ce, kd
        self.teacher.requires_grad_(False).eval()

    def get_extra_parameters(self):
        return 0

    def forward_train(self, image, target, epoch=0):
        logits_s, _ = self.student(image)
        with torch.no_grad():
            logits_t, _ = self.teacher(image)
        with torch.autocast(device_type=image.device.type, enabled=False):
            losses = {'loss_ce': self.ce_loss_weight * F.cross_entropy(logits_s.float(), target),
                      'loss_kd': self.kd_loss_weight * skd_loss(logits_s.float(), logits_t.float(),
                                                             self.temperature, self.tik, self.sqrt_floor)}
        return logits_s, logits_t, losses

    def forward_test(self, image):
        return self.student(image)[0]


__all__ = ['SKD', 'skd_loss']
