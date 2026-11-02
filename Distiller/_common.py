import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvReg(nn.Module):
    '''
    Convolutional Regression
    '''
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplementedError("Unsupported student/teacher feature shapes")
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)

def get_feat_shapes(student, teacher, input_size):
    # Shape probing must not consume RNG or update BatchNorm running buffers.
    states = [(module, module.training) for model in (student, teacher) for module in model.modules()]
    try:
        student.eval()
        teacher.eval()
        with torch.no_grad():
            _, feat_s = student(torch.zeros(1, 3, *input_size, device=next(student.parameters()).device))
            _, feat_t = teacher(torch.zeros(1, 3, *input_size, device=next(teacher.parameters()).device))
    finally:
        for module, training in states:
            module.training = training
    feat_s_shapes = [f.shape for f in feat_s["feats"]]
    feat_t_shapes = [f.shape for f in feat_t["feats"]]
    return feat_s_shapes, feat_t_shapes
