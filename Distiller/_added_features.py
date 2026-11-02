"""Architecture-aware stage extraction, excluding stems and pooled duplicates."""
import torch


def stage_indices(name, features):
    values = features['feats']
    candidates = range(len(values)) if name == 'mobilenetv1' else range(1, len(values))
    return [i for i in candidates if values[i].ndim == 4 and min(values[i].shape[-2:]) > 1]


def stages(name, features, preact=False):
    key = 'preact_feats' if preact else 'feats'
    return [features[key][i] for i in stage_indices(name, features)]


def probe(student, teacher):
    states = [(m, m.training) for model in (student, teacher) for m in model.modules()]
    try:
        student.eval()
        teacher.eval()
        with torch.no_grad():
            _, fs = student(torch.zeros(2, 3, 32, 32, device=next(student.parameters()).device, dtype=next(student.parameters()).dtype))
            _, ft = teacher(torch.zeros(2, 3, 32, 32, device=next(teacher.parameters()).device, dtype=next(teacher.parameters()).dtype))
        return fs, ft
    finally:
        for module, state in states:
            module.training = state
