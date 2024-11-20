import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Residual Block definition (used in both Perception and Throwing)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

# Perception module
class PerceptionModule(nn.Module):
    def __init__(self):
        super(PerceptionModule, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)  # C(3,64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # MP
        self.rb1 = ResidualBlock(128)  # RB(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1)
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # MP
        self.rb2 = ResidualBlock(256)  # RB(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=1)
        self.rb3 = ResidualBlock(512)  # RB(512)

    def forward(self, x):
        print(x.shape)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.mp1(x)
        x = self.rb1(x)
        x = F.relu(self.conv3(x))
        x = self.mp2(x)
        x = self.rb2(x)
        x = F.relu(self.conv4(x))
        x = self.rb3(x)
        return x

# Throwing module
class ThrowingModule(nn.Module):
    def __init__(self):
        super(ThrowingModule, self).__init__()
        self.rb1 = ResidualBlock(512)  # RB(256)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.rb2 = ResidualBlock(256)  # RB(128)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # UP
        self.conv3 = nn.Conv2d(256, 128, kernel_size=1)
        self.rb3 = ResidualBlock(128)   # RB(64)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # UP
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1)  # C(1,1)
        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.rb1(x)
        x = self.conv2(x)
        x = self.rb2(x)
        x = self.upsample1(x)
        x = self.conv3(x)
        print(x.shape)
        x = self.rb3(x)
        x = self.upsample2(x)
        x = self.conv1(x)
        x = self.conv_out(x)
        return x

# Combined model with Perception and Throwing
class PerceptionThrowingModel(nn.Module):
    def __init__(self):
        super(PerceptionThrowingModel, self).__init__()
        self.perception = PerceptionModule()
        self.throwing = ThrowingModule()

    def forward(self, x, velocity):
        x = self.perception(x)  # Pass through Perception module
        batch_size, filters, height, width = x.size()
        velocity = torch.tensor(velocity, dtype=torch.float32)  # Convert velocity to tensor
        v_tensor = np.full((1, filters, height, width), velocity.item())
# Concatenate `v_tensor` with `feature_representation` along the channel dimension
# This will result in a tensor of shape [batch_size, 256, height, width]
        print(v_tensor.shape)
        concatenated_features = x+velocity
        print(concatenated_features.shape)
        #concatenated_features = torch.cat((x, v_tensor), dim=1)
        x = self.throwing(concatenated_features)    # Pass through Throwing module
        return x

