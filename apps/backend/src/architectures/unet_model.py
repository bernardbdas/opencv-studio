"""Customizable U-Net PyTorch Architecture with CUDA acceleration support for opencv-studio."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from apps.backend.src.cuda.cuda_extension import run_fused_unet_activation

class FusedActivationBlock(nn.Module):
    """Custom block using fused CUDA activation kernel with PyTorch fallback."""
    def __init__(self, use_cuda_kernel=True):
        super().__init__()
        self.use_cuda_kernel = use_cuda_kernel

    def forward(self, x):
        if self.use_cuda_kernel:
            return run_fused_unet_activation(x)
        return F.relu(x, inplace=True)

class DoubleConv(nn.Module):
    """(Convolution -> Batch Normalization -> Activation) * 2"""
    def __init__(self, in_channels: int, out_channels: int, use_cuda_kernel: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            FusedActivationBlock(use_cuda_kernel=use_cuda_kernel),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            FusedActivationBlock(use_cuda_kernel=use_cuda_kernel),
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """Attention Gate for highlighting salient features passed through skip connections."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        net = self.relu(g1 + x1)
        out = self.psi(net)
        return x * out

class ConfigurableUNet(nn.Module):
    """Dynamic, fully customizable U-Net architecture.
    
    Supports:
    - Customizable depth (2 to 5 levels)
    - Customizable base feature channels (e.g. 16, 32, 64)
    - Attention Gates toggle
    - Custom CUDA kernel accelerated activation toggle
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        depth: int = 4,
        init_features: int = 32,
        use_attention: bool = True,
        use_cuda_kernel: bool = False
    ):
        super().__init__()
        self.depth = max(2, min(depth, 5))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.use_attention = use_attention
        self.use_cuda_kernel = use_cuda_kernel

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.attentions = nn.ModuleList()

        # Encoder Path
        curr_in = in_channels
        curr_out = init_features
        features = []
        for i in range(self.depth):
            self.encoders.append(DoubleConv(curr_in, curr_out, use_cuda_kernel=use_cuda_kernel))
            features.append(curr_out)
            if i < self.depth - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
                curr_in = curr_out
                curr_out = curr_out * 2

        # Bottleneck (bottom of U-Net)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, use_cuda_kernel=use_cuda_kernel)
        bottleneck_out = features[-1] * 2

        # Decoder Path
        prev_channels = bottleneck_out
        for i in reversed(range(self.depth)):
            skip_channels = features[i]
            self.upconvs.append(nn.ConvTranspose2d(prev_channels, skip_channels, kernel_size=2, stride=2))
            if self.use_attention:
                self.attentions.append(AttentionGate(F_g=skip_channels, F_l=skip_channels, F_int=skip_channels // 2))
            else:
                self.attentions.append(nn.Identity())
            self.decoders.append(DoubleConv(skip_channels * 2, skip_channels, use_cuda_kernel=use_cuda_kernel))
            prev_channels = skip_channels

        # Final Segmentation Output Conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder forward pass
        out = x
        for i in range(self.depth):
            out = self.encoders[i](out)
            skip_connections.append(out)
            if i < self.depth - 1:
                out = self.pools[i](out)

        # Bottleneck forward pass
        out = self.bottleneck(out)

        # Decoder forward pass
        skip_connections = skip_connections[::-1]
        for i in range(self.depth):
            out = self.upconvs[i](out)
            skip = skip_connections[i]
            
            # Match spatial sizes if padding differs
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=True)
                
            if self.use_attention:
                skip = self.attentions[i](out, skip)
                
            out = torch.cat([skip, out], dim=1)
            out = self.decoders[i](out)

        logits = self.final_conv(out)
        return logits

    def get_summary(self):
        """Returns metadata summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "depth": self.depth,
            "init_features": self.init_features,
            "use_attention": self.use_attention,
            "use_cuda_kernel": self.use_cuda_kernel,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

    def load_custom_weights(self, weights_bytes: bytes) -> dict:
        """Loads custom PyTorch state_dict checkpoint from raw bytes."""
        import io
        buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(buffer, map_location="cpu")
        
        # Handle state_dict if wrapped in dict with 'state_dict' or 'model' key
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
            
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        return {
            "status": "success",
            "loaded_tensors": len(state_dict),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys
        }

    def export_weights_bytes(self) -> bytes:
        """Serializes current model state_dict into bytes for downloading."""
        import io
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.getvalue()

