"""Logits knowledge distillation, with the project's temperature convention."""
import torch
from torch.nn import functional as F
from .distiller import Distiller


def kd_loss(student, teacher, temperature=3.0):
    return F.kl_div(F.log_softmax(student / temperature, dim=1),
                    F.softmax(teacher.detach() / temperature, dim=1),
                    reduction="batchmean") * temperature ** 2


class KD(Distiller):
    def __init__(self, student, teacher, ce=1.0, distill=16.0, temperature=3.0):
        super().__init__(student, teacher)
        self.ce, self.distill, self.temperature = ce, distill, temperature

    def forward_train(self, image, target, epoch=0):
        logits, _ = self.student(image)
        with torch.no_grad():
            teacher_logits, _ = self.teacher(image)
        return logits, teacher_logits, {"loss_ce": self.ce * F.cross_entropy(logits, target),
                                       "loss_kd": self.distill * kd_loss(logits, teacher_logits, self.temperature)}
