"""P2 "Audytor modeli": the network every model in the data set uses (copied from example/model.py).

    from lab_2.model import load
    net = load("test/models/3fa9c01b77de.pt")        # eval mode, float32, on CPU
    logits = net(x)                                   # x: (N, 3, 64, 64), values in [0, 1]

Normalisation is inside the network. The output layer is a cosine classifier: logits = 16 * cos(feature, row).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

N_CLASSES = 20


class CosineLinear(nn.Module):
    """Output layer whose logits are scaled cosine similarities between features and weight rows.

    Only the direction of each weight row matters, not its length.
    """

    def __init__(self, d, n, scale=16.0):
        """Initialises random weight rows.

        Args:
            d: Feature dimension.
            n: Number of classes.
            scale: Constant multiplier of the cosine similarities.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, d) / d**0.5)
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x):
        """Computes the logits.

        Args:
            x: Features (N, d).

        Returns:
            Logits (N, n), each in [-scale, scale].
        """
        return self.scale * F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))


class Net(nn.Module):
    """ResNet-18 for 64 x 64 images with input normalisation and a cosine output layer."""

    def __init__(self, n=N_CLASSES):
        """Builds the network.

        Args:
            n: Number of output classes.
        """
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        r = torchvision.models.resnet18(num_classes=n)
        # small images: a 3x3 stride-1 stem and no max-pool keep the spatial resolution high
        r.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        r.maxpool = nn.Identity()
        r.fc = CosineLinear(r.fc.in_features, n)
        self.r = r

    def forward(self, x):
        """Classifies a batch of images.

        Args:
            x: Images (N, 3, 64, 64) with values in [0, 1]; normalisation is done here.

        Returns:
            Logits (N, n_classes).
        """
        return self.r((x - self.mean) / self.std)


def load(path, device="cpu"):
    """Loads a saved model.

    Args:
        path: Path to a `.pt` state dict (weights stored in float16).
        device: Device to move the network to.

    Returns:
        The network in eval mode with float32 weights.
    """
    net = Net()
    sd = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return net.to(device).eval()
