import torch
import torch.nn as nn
import math

def _make_freqs(n: int, mode: str = "linear", base: float = 2.0, gamma: float = 1.0):
    """
    Data range from -1 to 1 so better use 1 PI instead of 2 PI.
    Returns frequencies as a 1-D tensor.
    mode:
      - 'linear': π * [1, 2, ..., n] * gamma
      - 'exp':    π * [base^0, base^1, ..., base^(n-1)] * gamma
    """
    if n <= 0:
        return torch.empty(0, dtype=torch.float32)
    if mode == "linear": # use 1 PI because the input always range between [-1,1] to use tanh activation.
        k = torch.arange(1, n + 1, dtype=torch.float32)  # 1..n
        return (1.0 * math.pi) * k * gamma
    else:
        k = torch.arange(n, dtype=torch.float32)         # 0..n-1
        return (math.pi) * (torch.pow(torch.tensor(base, dtype=torch.float32), k)) * gamma

class PositionalEncoding_Geo(nn.Module):
    def __init__(self,
                 num_freqs_space=(6, 6, 6),
                 include_input=True,
                 mode: str = "linear",    # 'linear' or 'exp'
                 base: float = 2.0,
                 gamma_space: float = 1.0,
                 normalize_features: bool = True):
        super().__init__()
        self.include_input = include_input
        self.normalize_features = normalize_features

        fx = _make_freqs(num_freqs_space[0], mode, base, gamma_space)
        fy = _make_freqs(num_freqs_space[1], mode, base, gamma_space)
        fz = _make_freqs(num_freqs_space[2], mode, base, gamma_space)

        self.register_buffer("freq_x", fx)
        self.register_buffer("freq_y", fy)
        self.register_buffer("freq_z", fz)

        # per-axis scaling so total variance stays ~1
        sx = (1.0 / math.sqrt(max(1, 2 * len(fx)))) if normalize_features and len(fx) > 0 else 1.0
        sy = (1.0 / math.sqrt(max(1, 2 * len(fy)))) if normalize_features and len(fy) > 0 else 1.0
        sz = (1.0 / math.sqrt(max(1, 2 * len(fz)))) if normalize_features and len(fz) > 0 else 1.0
        self.register_buffer("scale_x", torch.tensor(sx, dtype=torch.float32))
        self.register_buffer("scale_y", torch.tensor(sy, dtype=torch.float32))
        self.register_buffer("scale_z", torch.tensor(sz, dtype=torch.float32))

    def forward(self, X_train):
        # X: (N,3) with (x,y,z) normalized to [0,1] (outputs are in [-1,1])
        x, y, z = X_train[:, 0:1], X_train[:, 1:2], X_train[:, 2:3]
        out = [X_train] if self.include_input else []

        if self.freq_x.numel() > 0:
            sx = x * self.freq_x  # (N, Fx)
            out.append(self.scale_x * torch.sin(sx))
            out.append(self.scale_x * torch.cos(sx))
        if self.freq_y.numel() > 0:
            sy = y * self.freq_y
            out.append(self.scale_y * torch.sin(sy))
            out.append(self.scale_y * torch.cos(sy))
        if self.freq_z.numel() > 0:
            sz = z * self.freq_z
            out.append(self.scale_z * torch.sin(sz))
            out.append(self.scale_z * torch.cos(sz))
        return torch.cat(out, dim=-1) if len(out) > 0 else X_train

class PositionalEncoding_GeoTime(nn.Module):
    def __init__(self,
                 num_freqs_space=(6, 6, 6),
                 num_freqs_time=4,
                 include_input=True,
                 mode: str = "linear",
                 base: float = 2.0,
                 gamma_space: float = 1.0,
                 gamma_time: float = 1.0,
                 normalize_features: bool = True):
        super().__init__()
        self.include_input = include_input
        self.normalize_features = normalize_features

        fx = _make_freqs(num_freqs_space[0], mode, base, gamma_space)
        fy = _make_freqs(num_freqs_space[1], mode, base, gamma_space)
        fz = _make_freqs(num_freqs_space[2], mode, base, gamma_space)
        ft = _make_freqs(num_freqs_time,      mode, base, gamma_time)

        self.register_buffer("freq_x", fx)
        self.register_buffer("freq_y", fy)
        self.register_buffer("freq_z", fz)
        self.register_buffer("freq_t", ft)

        sx = (1.0 / math.sqrt(max(1, 2 * len(fx)))) if normalize_features and len(fx) > 0 else 1.0
        sy = (1.0 / math.sqrt(max(1, 2 * len(fy)))) if normalize_features and len(fy) > 0 else 1.0
        sz = (1.0 / math.sqrt(max(1, 2 * len(fz)))) if normalize_features and len(fz) > 0 else 1.0
        st = (1.0 / math.sqrt(max(1, 2 * len(ft)))) if normalize_features and len(ft) > 0 else 1.0
        self.register_buffer("scale_x", torch.tensor(sx, dtype=torch.float32))
        self.register_buffer("scale_y", torch.tensor(sy, dtype=torch.float32))
        self.register_buffer("scale_z", torch.tensor(sz, dtype=torch.float32))
        self.register_buffer("scale_t", torch.tensor(st, dtype=torch.float32))

    def forward(self, X_train):
        # print(X_train.shape)
        # X: (N,4) with (x,y,z,t) normalized to [0,1]
        x, y, z, t = X_train[:, 0:1], X_train[:, 1:2], X_train[:, 2:3], X_train[:, 3:4]
        out = [X_train] if self.include_input else []

        if self.freq_x.numel() > 0:
            sx = x * self.freq_x
            out.append(self.scale_x * torch.sin(sx)); out.append(self.scale_x * torch.cos(sx))
        if self.freq_y.numel() > 0:
            sy = y * self.freq_y
            out.append(self.scale_y * torch.sin(sy)); out.append(self.scale_y * torch.cos(sy))
        if self.freq_z.numel() > 0:
            sz = z * self.freq_z
            out.append(self.scale_z * torch.sin(sz)); out.append(self.scale_z * torch.cos(sz))
        if self.freq_t.numel() > 0:
            st = t * self.freq_t
            out.append(self.scale_t * torch.sin(st)); out.append(self.scale_t * torch.cos(st))
        return torch.cat(out, dim=-1) if len(out) > 0 else X_train
