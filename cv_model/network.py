import torch
import torch.nn as nn
import torch.nn.functional as F

# Define SE Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y

# Define Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            SEBlock(16)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            SEBlock(32)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            SEBlock(64)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# Primary Capsules
class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, capsule_dim, num_capsules, kernel_size=3, stride=2):
        super(PrimaryCapsules, self).__init__()
        self.capsules = nn.Conv2d(in_channels, num_capsules * capsule_dim,
                                  kernel_size=kernel_size, stride=stride, padding=1)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

    def forward(self, x):
        batch_size = x.size(0)
        u = self.capsules(x)
        u = u.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        u = u.permute(0, 1, 3, 2).contiguous()
        u = u.view(batch_size, -1, self.capsule_dim)
        return self.squash(u)

    def squash(self, s):
        s_norm = torch.norm(s, dim=-1, keepdim=True)
        return (s_norm**2 / (1 + s_norm**2)) * (s / (s_norm + 1e-8))

# Digit Capsules
class DigitCapsules(nn.Module):
    def __init__(self, input_capsules, input_dim, num_classes, capsule_dim, routing_iters=3):
        super(DigitCapsules, self).__init__()
        self.num_classes = num_classes
        self.routing_iters = routing_iters
        self.input_capsules = input_capsules
        self.W = nn.Parameter(0.01 * torch.randn(1, input_capsules, num_classes, capsule_dim, input_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        W = self.W.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W, x).squeeze(-1)
        b_ij = torch.zeros(batch_size, self.input_capsules, self.num_classes, 1, device=x.device)

        for _ in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, s):
        s_norm = torch.norm(s, dim=-1, keepdim=True)
        return (s_norm**2 / (1 + s_norm**2)) * (s / (s_norm + 1e-8))

# Full Model
class HR_SEMobileCapsNet(nn.Module):
    def __init__(self, num_classes=3):
        super(HR_SEMobileCapsNet, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.primary_capsules = PrimaryCapsules(
            in_channels=64,
            capsule_dim=8,
            num_capsules=8,
            kernel_size=3,
            stride=2
        )
        self.digit_capsules = DigitCapsules(
            input_capsules=8 * 16 * 16,
            input_dim=8,
            num_classes=num_classes,
            capsule_dim=16
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return torch.norm(x, dim=-1)

# Loader function
def load_model(model_path="cv_model/final.pth", device='cpu'):
    model = HR_SEMobileCapsNet(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model