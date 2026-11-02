import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
    
    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        '''
        self.children()은 Distiller 클래스의 자식 모듈들을 반복하여 접근
        즉, nn.Module 서브클래스들을 접근하는 것임
        nn.Module 서브클래스들에 접근하여 자식 모듈의 train() 메서드를 호출하여 훈련 모드로 설정
        그리고 self.teacher는 평가모드로 설정
        '''
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self
    
    def get_learnable_parameter(self):
        return [v for k, v in self.student.named_parameters()]
    
    def forward_train(self, **kwargs):
        raise NotImplementedError()
    
    def forward_test(self, image):
        # output, _ = self.student(image)
        output = self.student(image)
        return output
    
    '''
    CRD 때문에 임시시
    '''
    def forward(self, image, label, epoch=None):
        if self.training:
            return self.forward_train(image, label, epoch)
        return self.forward_test(image)

class Vanilla(nn.Module):
    '''
    가장 기본적인 형태의 KD, Vanilla : 기본적이고 간단한 형태를 나타내는 용어
    Teacher와 Student 두 모델의 출력(logits)을 얻어, KD 손실 계산.
    '''
    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student
    
    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]
    
    def forward_train(self, image, target, **kwargs):
        # logits_student, _ = self.student(image)
        logits_student = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, loss
    
    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])
    