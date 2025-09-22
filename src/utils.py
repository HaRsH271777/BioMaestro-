import torch
import torch.nn as nn
import torchaudio.transforms as T

class PCEN(nn.Module):
    """
    Per-Channel Energy Normalization (PCEN) implementation.
    This module is designed to be a differentiable layer in a PyTorch model.
    It enhances faint signals and suppresses loud, stationary noise.
    """
    def __init__(
        self,
        num_bands,
        s=0.025,
        alpha=0.98,
        delta=2.0,
        r=0.5,
        eps=1e-6
    ):
        super().__init__()
        self.s = s
        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.eps = eps
        
        # Trainable smoothing coefficient
        self.log_s = nn.Parameter(torch.log(torch.full((num_bands,), s)))
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input mel spectrogram of shape (B, C, F, T) or (B, F, T)
        
        Returns:
            torch.Tensor: PCEN-processed spectrogram.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1) # Add channel dimension if not present
        
        # CORRECTED LINE: Reshape s to be broadcastable with (B, C, F)
        s = torch.exp(self.log_s).to(x.device)[None, None, :]
        
        # Differentiable EMA smoother
        smoother = [x[..., 0]]
        for i in range(1, x.shape[-1]):
            smoother_val = (1 - s) * smoother[-1] + s * x[..., i]
            smoother.append(smoother_val)
        
        M = torch.stack(smoother, dim=-1)
        
        # PCEN calculation
        pcen_out = (x / (self.eps + M)**self.alpha + self.delta)**self.r - self.delta**self.r
        return pcen_out.squeeze(1)