import torchvision
from torchvision.models import resnet101, ResNet101_Weights

weights = torchvision.models.get_weight('MobileNet_V3_Large_Weights.IMAGENET1K_V1')
weights = torchvision.models.get_weight('ResNet101_Weights.IMAGENET1K_V1')

_ = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

