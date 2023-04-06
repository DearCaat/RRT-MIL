import torch.nn as nn
from torchvision import models
class Resnet(nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()

        self.model = list(models.resnet50(pretrained = True).children())[:-3]
        self.features = nn.Sequential(*self.model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_extractor_part2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=self.feature_extractor_part2(x)
        return x