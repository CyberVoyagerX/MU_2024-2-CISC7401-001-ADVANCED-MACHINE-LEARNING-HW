import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

# 定义 CIFAR-10 的类别数
N_CLASSES = 10

def get_resnet18_cifar(num_classes: int = N_CLASSES, pretrained: bool = False) -> nn.Module:
    """
    Loads a ResNet-18 model and modifies it for CIFAR-10.
    Args:
        num_classes: The number of output classes 
        pretrained: If True, attempts to load weights pretrained on ImageNet.
                           Note: Modifying the first layer invalidates its pretrained weights.
                           It's generally recommended to train from scratch for CIFAR-10
                           unless using specific transfer learning techniques.

    Returns:
        nn.Module: The modified ResNet-18 model.
    """
    print(f"Loading ResNet-18 {'with pretrained weights' if pretrained else 'from scratch'}.")

    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    model.maxpool = nn.Identity()

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)


    return model