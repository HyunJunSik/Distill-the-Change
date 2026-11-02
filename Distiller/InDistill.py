"""InDistill-KD stage-port; ordered warmup followed by CE and logits KD.

Uses the same target rules as distiller_additions_11. Heterogeneous pairs
require a prepared auxiliary teacher; construction never trains one silently.
"""
import torch
import torch.nn.functional as F

from .distiller import Distiller
from ._added_common import kd_loss, model_name
from ._added_features import probe
from ._indistill_core import InformationTargets, set_trainable_prefix, widen_student


class InDistill(Distiller):
    def __init__(self, student, teacher, student_name=None, teacher_name=None, *,
                 auxiliary_teacher=None, auxiliary_checkpoint=None,
                 ce=1., kd=16., temp=3., a=5, b=1, intermediate_weight=1.):
        if auxiliary_teacher is not None and auxiliary_checkpoint is not None:
            raise ValueError('Pass either auxiliary_teacher or auxiliary_checkpoint, not both.')
        student_name = model_name(student, student_name)
        if auxiliary_checkpoint is not None:
            auxiliary_teacher = widen_student(student)
            # Owner-provided checkpoint, including prepared_teacher.pt from the
            # 11-pair runner. This is input loading only when the user calls it.
            payload = torch.load(auxiliary_checkpoint, map_location='cpu', weights_only=False)
            auxiliary_teacher.load_state_dict(payload.get('model', payload), strict=True)
            auxiliary_teacher.to(next(student.parameters()).device)
        if auxiliary_teacher is not None:
            teacher = auxiliary_teacher
            teacher_name = student_name
        super().__init__(student, teacher)
        if temp <= 0 or min(ce, kd, intermediate_weight) < 0:
            raise ValueError('Positive temperature and nonnegative loss weights required.')
        self.student_name = student_name
        self.teacher_name = model_name(teacher, teacher_name)
        self.ce_loss_weight, self.kd_loss_weight = ce, kd
        self.temperature, self.intermediate_weight = temp, intermediate_weight
        self.teacher.requires_grad_(False).eval()
        fs, ft = probe(student, teacher)
        self.information = InformationTargets(teacher, self.teacher_name, self.student_name, fs, ft, a, b)
        self.information.to(next(student.parameters()).device)
        self.warmup_epochs = self.information.warmup_epochs
        self.feature_layout = self.information.layout

    def get_extra_parameters(self):
        return 0

    def auxiliary_state(self):
        return {'information': self.information.state_dict()}

    def load_auxiliary_state(self, state):
        self.information.load_state_dict(state['information'])

    def forward_train(self, image, target, epoch=0):
        stage = self.information.stage_for_epoch(epoch)
        set_trainable_prefix(self.student, self.student_name, stage)
        logits_s, fs = self.student(image)
        with torch.no_grad():
            logits_t, ft = self.teacher(image)
        with torch.autocast(device_type=image.device.type, enabled=False):
            if stage is not None:
                loss_ce = logits_s.detach().new_zeros((), dtype=torch.float32)
                loss_kd = self.intermediate_weight * self.information(fs, ft, len(target), epoch)
            else:
                loss_ce = self.ce_loss_weight * F.cross_entropy(logits_s.float(), target)
                loss_kd = self.kd_loss_weight * kd_loss(logits_s.float(), logits_t.float(), self.temperature)
        return logits_s, logits_t, {'loss_ce': loss_ce, 'loss_kd': loss_kd}

    def forward_test(self, image):
        return self.student(image)[0]


__all__ = ['InDistill', 'InformationTargets', 'widen_student']
