"""Small independent helpers for the three added baseline distillers."""
import torch.nn.functional as F


def kd_loss(student, teacher, temperature):
    return F.kl_div(F.log_softmax(student / temperature, dim=1),
                    F.softmax(teacher.detach() / temperature, dim=1),
                    reduction='batchmean') * temperature ** 2


def model_name(model, explicit=None):
    if explicit is not None:
        key = explicit.lower().replace('_', '').replace('-', '')
        aliases = {'shufflev1': 'ShuffleV1', 'shufflenetv1': 'ShuffleV1',
                   'shufflev2': 'ShuffleV2', 'shufflenetv2': 'ShuffleV2',
                   'mobilenetv1': 'mobilenetv1'}
        return aliases.get(key, explicit.lower())
    name = type(model).__name__.lower()
    if 'mobilenetv1' in name:
        return 'mobilenetv1'
    if 'shuffle' in name:
        return 'ShuffleV2' if 'v2' in name else 'ShuffleV1'
    if hasattr(model, 'block0'):
        return 'vgg'
    if hasattr(model, 'nChannels') and hasattr(model, 'block3'):
        return 'wrn'
    if hasattr(model, 'layer1'):
        return 'ResNet50' if hasattr(model, 'layer4') else 'resnet'
    raise ValueError('Pass student_name/teacher_name explicitly for this model.')
