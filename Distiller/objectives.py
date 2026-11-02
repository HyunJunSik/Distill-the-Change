"""Baseline distillation on original images; optional complete ALTER training.

Equations and component implementations are ported from the project snapshots.
MLKD means MLKD_align_3; InDistill is the documented classification stage-port.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .KD import kd_loss
from .DKD import dkd_loss
from .RKD import rkd_loss
from .MLKD import kd_loss as mlkd_kl, cc_loss, bc_loss
from .FitNet import hint_regressor
from ._review_core import ReviewLoss
from ._skd_core import skd_loss
from ._indistill_core import InformationTargets, set_trainable_prefix
from .ALTER import MildRandomAffine, ResponseCompetenceCurriculum, elastic_log_strain_geometry_loss

METHODS = ("KD", "RKD", "ReviewKD", "InDistill", "DKD", "MLKD", "SKD", "FitNet")
DEFAULT_WEIGHTS = {"KD": (1., 16.), "RKD": (1., .25), "ReviewKD": (1., 1.),
                   "InDistill": (1., 16.), "DKD": (2., 1.), "MLKD": (.5, 8.),
                   "SKD": (.1, .9), "FitNet": (2., 100.), "CE": (1., 0.)}


class DistillationObjective(nn.Module):
    def __init__(self, student, teacher, method, *, alter=False, student_name="resnet8x4",
                 teacher_name="resnet32x4", input_size=32, ce=None, distill=None, response=200.):
        super().__init__()
        if method not in METHODS + ("CE",):
            raise ValueError(method)
        if method != "CE" and teacher is None:
            raise ValueError("A teacher checkpoint is required for distillation")
        self.student, self.teacher, self.method, self.alter = student, teacher, method, alter
        self.student_name, self.teacher_name = student_name, teacher_name
        weights = DEFAULT_WEIGHTS[method]
        self.ce = weights[0] if ce is None else ce
        self.distill = weights[1] if distill is None else distill
        self.response_weight = response
        self.extra = nn.ModuleDict()
        self.transform = MildRandomAffine(.95, 1.05)
        self.curriculum = ResponseCompetenceCurriculum()
        self.layout = {}
        if teacher is not None:
            teacher.requires_grad_(False).eval()
        if method in ("FitNet", "ReviewKD", "InDistill"):
            states = [(module, module.training) for net in (student, teacher) for module in net.modules()]
            try:
                student.eval()
                teacher.eval()
                with torch.no_grad():
                    sample = torch.zeros(2, 3, input_size, input_size, device=next(student.parameters()).device)
                    _, fs = student(sample)
                    _, ft = teacher(sample)
            finally:
                for module, training in states:
                    module.training = training
            if method == "FitNet":
                self.extra["hint"] = hint_regressor(fs["feats"][2], ft["feats"][2])
            elif method == "ReviewKD":
                self.extra["review"] = ReviewLoss(student_name, teacher_name, fs, ft, 512)
                self.layout = self.extra["review"].layout
            else:
                if input_size != 32:
                    raise ValueError("This release's InDistill stage-port supports CIFAR-100; ImageNet is not silently substituted")
                self.extra["information"] = InformationTargets(teacher, teacher_name, student_name, fs, ft, 5, 1)
                self.layout = self.extra["information"].layout

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def learnable_parameters(self):
        # Include initially frozen prefixes, as InDistill unfreezes them later.
        return list(self.student.parameters()) + list(self.extra.parameters())

    def auxiliary_state(self):
        return {"extra": self.extra.state_dict(), "curriculum": self.curriculum.state_dict()}

    def load_auxiliary_state(self, state):
        self.extra.load_state_dict(state["extra"])
        self.curriculum.load_state_dict(state["curriculum"])

    def native_loss(self, s, t, fs, ft, target, epoch):
        n, method = len(target), self.method
        s, t = s[:n].float(), t[:n].float()
        if method in ("KD", "InDistill"):
            if method == "InDistill" and self.extra["information"].stage_for_epoch(epoch) is not None:
                return self.extra["information"](fs, ft, n, epoch)
            return self.distill * kd_loss(s, t, 3.)
        if method == "DKD":
            return self.distill * min(epoch / 20., 1.) * dkd_loss(s, t, target, 3., 6., 4.)
        if method == "RKD":
            return self.distill * rkd_loss(fs["pooled_feat"][:n].float(), ft["pooled_feat"][:n].float())
        if method == "FitNet":
            return self.distill * F.mse_loss(self.extra["hint"](fs["feats"][2][:n].float()), ft["feats"][2][:n].float())
        if method == "ReviewKD":
            return self.distill * min(epoch / 20., 1.) * self.extra["review"](fs, ft, n)
        if method == "SKD":
            return self.distill * skd_loss(s, t, 4., .1, 1e-12)
        if method == "MLKD":
            probability = F.softmax(t.detach(), dim=1)
            confidence = probability.max(1).values
            class_confidence = probability.sum(0)
            mask = confidence.le(np.percentile(confidence.cpu().numpy(), 50))
            class_mask = class_confidence.le(np.percentile(class_confidence.cpu().numpy(), 50))
            return self.distill * sum(sum((fn(s, t, temp) * selection).mean()
                                          for temp in (4., 3., 2.))
                                     for fn, selection in ((mlkd_kl, mask), (cc_loss, class_mask), (bc_loss, mask)))
        raise ValueError(method)

    def forward(self, images, target, epoch=0):
        warmup = self.method == "InDistill" and self.extra["information"].stage_for_epoch(epoch) is not None
        if self.method == "InDistill":
            set_trainable_prefix(self.student, self.student_name, self.extra["information"].stage_for_epoch(epoch))
        use_alter = self.alter and not warmup
        n = len(target)
        severity = self.curriculum.severity() if use_alter else images.new_tensor(.25)
        inputs = torch.cat((images, self.transform(images, severity)), 0) if use_alter else images
        logits, fs = self.student(inputs)
        base_ce = F.cross_entropy(logits[:n].float(), target, reduction="none")
        ce = base_ce.mean()
        losses = {}
        metrics = {"severity": float(severity), "augmented_images": n if use_alter else 0}
        if self.method != "CE":
            with torch.no_grad():
                teacher_logits, ft = self.teacher(inputs)
            losses["distillation"] = self.native_loss(logits, teacher_logits, fs, ft, target, epoch)
        if use_alter:
            augmented_ce = F.cross_entropy(logits[n:].float(), target, reduction="none")
            ce = .5 * (ce + augmented_ce.mean())
            response, matching = elastic_log_strain_geometry_loss(fs["feats"][-1][:n], fs["feats"][-1][n:],
                                                                  ft["feats"][-1][:n], ft["feats"][-1][n:], eps=1e-5)
            losses["response"] = self.response_weight * response
            with torch.no_grad():
                pb, pa = torch.exp(-base_ce), torch.exp(-augmented_ce)
                task_competence = (2 * pb * pa / (pb + pa + 1e-5)).mean()
                self.curriculum.update(matching, task_competence)
            metrics.update(response_matching=float(matching.mean()), task_competence=float(task_competence))
        losses["ce"] = ce.detach() * 0 if warmup else self.ce * ce
        return logits[:n], losses, metrics
