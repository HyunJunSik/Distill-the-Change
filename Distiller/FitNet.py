"""FitNet hint mapping with the project's resize fallback."""
import torch
from torch import nn
from torch.nn import functional as F
from .distiller import Distiller
from ._common import ConvReg
from ._added_features import probe


class ResizeConvReg(nn.Module):
    def __init__(self, student_shape, teacher_shape):
        super().__init__()
        self.size = tuple(teacher_shape[-2:])
        self.mapping = nn.Sequential(nn.Conv2d(student_shape[1], teacher_shape[1], 1),
                                     nn.BatchNorm2d(teacher_shape[1]), nn.ReLU(inplace=True))

    def forward(self, feature):
        return self.mapping(F.interpolate(feature, size=self.size, mode="bilinear", align_corners=False))


def hint_regressor(student_feature, teacher_feature):
    try:
        return ConvReg(student_feature.shape, teacher_feature.shape)
    except NotImplementedError:
        return ResizeConvReg(student_feature.shape, teacher_feature.shape)


class FitNet(Distiller):
    def __init__(self, student, teacher, ce=2.0, distill=100.0):
        super().__init__(student, teacher)
        fs, ft = probe(student, teacher)
        self.regressor = hint_regressor(fs["feats"][2], ft["feats"][2])
        self.ce, self.distill = ce, distill

    def get_learnable_parameter(self):
        return list(self.student.parameters()) + list(self.regressor.parameters())

    def forward_train(self, image, target, epoch=0):
        logits, fs = self.student(image)
        with torch.no_grad():
            teacher_logits, ft = self.teacher(image)
        return logits, teacher_logits, {"loss_ce": self.ce * F.cross_entropy(logits, target),
            "loss_kd": self.distill * F.mse_loss(self.regressor(fs["feats"][2]), ft["feats"][2])}
