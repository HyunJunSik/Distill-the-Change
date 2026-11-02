"""Dynamic-shape ReviewKD with the repository's legacy distiller interface."""
import torch
import torch.nn.functional as F

from .distiller import Distiller
from ._added_common import model_name
from ._added_features import probe
from ._review_core import ABF, ReviewLoss, hcl_loss


class ReviewKD(Distiller):
    def __init__(self, student, teacher, student_name=None, teacher_name=None,
                 *, ce=1., review=1., warmup_epochs=20, max_mid_channel=512):
        super().__init__(student, teacher)
        if warmup_epochs <= 0:
            raise ValueError('warmup_epochs must be positive.')
        self.student_name = model_name(student, student_name)
        self.teacher_name = model_name(teacher, teacher_name)
        self.ce_loss_weight, self.reviewkd_loss_weight = ce, review
        self.warmup_epochs = warmup_epochs
        self.teacher.requires_grad_(False).eval()
        fs, ft = probe(student, teacher)
        self.review_loss = ReviewLoss(self.student_name, self.teacher_name, fs, ft, max_mid_channel)
        self.review_loss.to(device=next(student.parameters()).device, dtype=torch.float32)
        self.feature_layout = self.review_loss.layout

    def get_learnable_parameter(self):
        return super().get_learnable_parameter() + list(self.review_loss.parameters())

    def get_extra_parameters(self):
        return sum(p.numel() for p in self.review_loss.parameters())

    def auxiliary_state(self):
        return {'review': self.review_loss.state_dict()}

    def load_auxiliary_state(self, state):
        self.review_loss.load_state_dict(state['review'])

    def forward_train(self, image, target, epoch=0):
        logits_s, fs = self.student(image)
        with torch.no_grad():
            logits_t, ft = self.teacher(image)
        with torch.autocast(device_type=image.device.type, enabled=False):
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_s.float(), target)
            loss_kd = self.reviewkd_loss_weight * min(epoch / self.warmup_epochs, 1.) * self.review_loss(fs, ft, len(target))
        return logits_s, logits_t, {'loss_ce': loss_ce, 'loss_kd': loss_kd}

    def forward_test(self, image):
        return self.student(image)[0]


__all__ = ['ReviewKD', 'ABF', 'ReviewLoss', 'hcl_loss']
