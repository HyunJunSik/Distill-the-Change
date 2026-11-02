import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from .distiller import Distiller

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def _autocast_disabled(tensor):
    if tensor.device.type in {"cpu", "cuda", "xpu"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def cnn_feature_descriptor(feature):
    with _autocast_disabled(feature):
        descriptor = feature.float().mean(dim=(-2, -1))
        return F.normalize(descriptor, p=2.0, dim=1, eps=1e-12)


def cnn_elastic_response(feature_base, feature_augmented, eps=1e-6):
    """Return log-strain, self-response, and the base-view Gram matrix."""

    descriptor_base = cnn_feature_descriptor(feature_base)
    descriptor_augmented = cnn_feature_descriptor(feature_augmented)

    with _autocast_disabled(descriptor_base):
        relation_base = (
            descriptor_base @ descriptor_base.transpose(0, 1)
        ).clamp(-1.0, 1.0)
        relation_augmented = (
            descriptor_augmented @ descriptor_augmented.transpose(0, 1)
        ).clamp(-1.0, 1.0)

        distance_sq_base = (
            2.0 - 2.0 * relation_base
        ).clamp_min(float(eps))
        distance_sq_augmented = (
            2.0 - 2.0 * relation_augmented
        ).clamp_min(float(eps))

        log_strain = 0.5 * (
            torch.log(distance_sq_augmented)
            - torch.log(distance_sq_base)
        )
        self_response = 1.0 - torch.sum(
            descriptor_base * descriptor_augmented,
            dim=1,
        ).clamp(-1.0, 1.0)

    return log_strain, self_response, relation_base


def elastic_log_strain_geometry_loss(
    feature_student_base,
    feature_student_augmented,
    feature_teacher_base,
    feature_teacher_augmented,
    eps=1e-6,
):
    """
    Match transformation-induced elastic geometry deformation.

    The function returns the batch loss and one detached geometry-agreement
    score per sample. Agreement is scale-free and lies in [0, 1].
    """

    strain_student, self_student, _ = cnn_elastic_response(
        feature_student_base,
        feature_student_augmented,
        eps,
    )

    with torch.no_grad():
        strain_teacher, self_teacher, relation_teacher_base = (
            cnn_elastic_response(
                feature_teacher_base,
                feature_teacher_augmented,
                eps,
            )
        )

    batch_size = strain_student.shape[0]
    off_diagonal = ~torch.eye(
        batch_size,
        dtype=torch.bool,
        device=strain_student.device,
    )

    with _autocast_disabled(strain_student):
        with torch.no_grad():
            teacher_distance = torch.sqrt(
                (
                    2.0 - 2.0 * relation_teacher_base
                ).clamp_min(float(eps))
            )
            off_diagonal_distance = teacher_distance[off_diagonal]
            distance_scale = (
                off_diagonal_distance.median()
                if off_diagonal_distance.numel() > 0
                else teacher_distance.new_tensor(1.0)
            ).clamp_min(float(eps))
            edge_weight = torch.exp(
                -0.5 * (teacher_distance / distance_scale).square()
            )
            edge_weight = edge_weight * off_diagonal.to(edge_weight.dtype)

        edge_weight_sum = edge_weight.sum(dim=1).clamp_min(float(eps))
        strain_difference = strain_student - strain_teacher
        strain_loss_per_sample = (
            edge_weight * strain_difference.square()
        ).sum(dim=1) / edge_weight_sum
        self_loss_per_sample = (self_student - self_teacher).square()

        loss_per_sample = (
            0.5 * self_loss_per_sample
            + 0.5 * strain_loss_per_sample
        )

        with torch.no_grad():
            strain_error = torch.sqrt(strain_loss_per_sample.detach())
            strain_student_norm = torch.sqrt(
                (
                    edge_weight * strain_student.detach().square()
                ).sum(dim=1) / edge_weight_sum
            )
            strain_teacher_norm = torch.sqrt(
                (
                    edge_weight * strain_teacher.square()
                ).sum(dim=1) / edge_weight_sum
            )
            strain_agreement = 1.0 - strain_error / (
                strain_student_norm
                + strain_teacher_norm
                + float(eps)
            )

            self_error = (self_student.detach() - self_teacher).abs()
            self_agreement = 1.0 - self_error / (
                self_student.detach().abs()
                + self_teacher.abs()
                + float(eps)
            )

            geometry_agreement = (
                0.5 * strain_agreement
                + 0.5 * self_agreement
            ).clamp(0.0, 1.0)

    return loss_per_sample.mean(), geometry_agreement.detach()


class MildRandomAffine(nn.Module):
    def __init__(self, min_scale=0.95, max_scale=1.05):
        super().__init__()
        self.degrees = 7.0 # 5.0이 default
        self.translate = 0.07 # 0.05가 default
        self.scale = (min_scale, max_scale)

    def forward(self, image, severity):
        batch_size = image.shape[0]
        device = image.device
        severity = torch.as_tensor(
            severity,
            device=device,
            dtype=torch.float32,
        ).detach().flatten()[0].clamp(0.0, 1.0)

        angles = (
            torch.rand(batch_size, device=device) * 2.0 - 1.0
        ) * (severity * self.degrees * torch.pi / 180.0)

        scale_min = 1.0 + severity * (self.scale[0] - 1.0)
        scale_max = 1.0 + severity * (self.scale[1] - 1.0)
        scales = scale_min + torch.rand(
            batch_size,
            device=device,
        ) * (scale_max - scale_min)

        translations = (
            torch.rand(batch_size, 2, device=device) * 2.0 - 1.0
        ) * (severity * self.translate)

        cosine = torch.cos(angles) / scales
        sine = torch.sin(angles) / scales
        theta = torch.zeros(
            batch_size,
            2,
            3,
            device=device,
            dtype=torch.float32,
        )
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine
        theta[:, 1, 0] = sine
        theta[:, 1, 1] = cosine
        theta[:, :, 2] = translations

        with _autocast_disabled(image):
            image_fp32 = image.float()
            grid = F.affine_grid(
                theta,
                image_fp32.shape,
                align_corners=False,
            )
            augmented = F.grid_sample(
                image_fp32,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            # Gaussian Blur : idependently applied to each sample
            kernel = augmented.new_tensor([
                [1.0, 2.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 2.0, 1.0],
            ]) / 16.0

            kernel = kernel.view(1, 1, 3, 3).repeat(
                augmented.shape[1], 1, 1, 1
            )
            blurred = F.conv2d(
                F.pad(augmented, (1, 1, 1, 1), mode="reflect"),
                kernel,
                groups=augmented.shape[1],
            )
            
            blur_mask = (
                torch.rand(batch_size, 1, 1, 1, device=augmented.device,) < 0.5
            ).to(augmented.dtype)

            augmented = torch.lerp(augmented, blurred, severity * blur_mask)

        return augmented.to(dtype=image.dtype)


class ResponseCompetenceCurriculum(nn.Module):
    """EMA-smoothed student readiness controlling affine severity."""

    MIN_SEVERITY = 0.25
    MAX_SEVERITY = 1.0
    TASK_BALANCE_FACTOR = 1.10

    def __init__(self):
        super().__init__()
        self.beta = 0.99
        self.register_buffer(
            "instant_geometry_agreement",
            torch.tensor(0.0)
        )
        self.register_buffer(
            "instant_task_readiness",
            torch.tensor(0.0)
        )
        self.register_buffer("instant_competence", torch.tensor(0.0))
        self.register_buffer(
            "ema_geometry_agreement",
            torch.tensor(0.0)
        )
        self.register_buffer(
            "ema_task_readiness",
            torch.tensor(0.0),
        )
        self.register_buffer(
            "ema_competence",
            torch.tensor(0.0),
        )
        self.register_buffer(
            "current_severity",
            torch.tensor(self.MIN_SEVERITY),
        )
        self.register_buffer(
            "last_severity",
            torch.tensor(self.MIN_SEVERITY),
        )

    @torch.no_grad()
    def severity(self):
        self.last_severity.copy_(self.current_severity)
        return self.current_severity.detach().clone()

    @torch.no_grad()
    def update(self, geometry_agreement, task_readiness):
        geometry_score = (geometry_agreement.detach().float().mean().clamp(0.0, 1.0))
        task_score = (task_readiness.detach().float().mean().clamp(0.0, 1.0))

        self.instant_geometry_agreement.copy_(geometry_score)
        self.instant_task_readiness.copy_(task_score)
        instant_competence = torch.minimum(
            geometry_score, task_score,
        )
        self.instant_competence.copy_(
            instant_competence
        )
        # ver.3 Geometry와 task를 각각 독립적으로 EMA
        self.ema_geometry_agreement.mul_(
            self.beta
        ).add_(geometry_score, alpha=1.0 - self.beta)
        self.ema_task_readiness.mul_(
            self.beta
        ).add_(task_score, alpha=1.0-self.beta)

        # Calibrate the systematically smaller task-readiness scale only
        # when selecting the bottleneck. Keep its raw EMA unchanged.
        balanced_ema_task = (
            self.TASK_BALANCE_FACTOR * self.ema_task_readiness
        ).clamp(0.0, 1.0)

        # ver.3 평활화된 geometry, task 중에서 부족한 쪽을 최종 competence로 사용
        ema_competence = torch.minimum(
            self.ema_geometry_agreement,
            balanced_ema_task,
        )
        self.ema_competence.copy_(ema_competence)
        
        self.current_severity.copy_(
            self.MIN_SEVERITY
            + (self.MAX_SEVERITY - self.MIN_SEVERITY)
            * self.ema_competence
        )



