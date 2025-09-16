import torch
import torch.nn as nn
import torch.nn.functional as F

def _match_size_like(src, ref):
    sd, sh, sw = src.shape[-3:]
    rd, rh, rw = ref.shape[-3:]
    dz, dy, dx = max(0, sd - rd), max(0, sh - rh), max(0, sw - rw)
    if dz or dy or dx:
        z0, y0, x0 = dz // 2, dy // 2, dx // 2
        src = src[..., z0:sd - (dz - z0), y0:sh - (dy - y0), x0:sw - (dx - x0)]
    sd, sh, sw = src.shape[-3:]
    pd, ph, pw = max(0, rd - sd), max(0, rh - sh), max(0, rw - sw)
    if pd or ph or pw:
        src = F.pad(src, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2))
    return src

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.proj = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else None
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        i = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.proj is not None:
            i = self.proj(i)
        return self.act2(x + i)

class MultiScaleRoutedResUNet3D_DS(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=32, upsample_to_input=True):
        super().__init__()
        f = base_features
        self.default_upsample_to_input = bool(upsample_to_input)
        self.enc0 = ResBlock3D(in_channels, f)
        self.enc1 = ResBlock3D(f, f * 2)
        self.enc2 = ResBlock3D(f * 2, f * 4)
        self.enc3 = ResBlock3D(f * 4, f * 8)
        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = ResBlock3D(f * 8, f * 16)
        self.up3 = nn.ConvTranspose3d(f * 16, f * 8, 2, 2)
        self.dec3 = ResBlock3D(f * 16, f * 8)
        self.up2 = nn.ConvTranspose3d(f * 8, f * 4, 2, 2)
        self.dec2 = ResBlock3D(f * 8, f * 4)
        self.up1 = nn.ConvTranspose3d(f * 4, f * 2, 2, 2)
        self.dec1 = ResBlock3D(f * 4, f * 2)
        self.up0 = nn.ConvTranspose3d(f * 2, f, 2, 2)
        self.dec0 = ResBlock3D(f * 2, f)
        self.stem1 = ResBlock3D(in_channels, f * 2)
        self.stem2 = ResBlock3D(in_channels, f * 4)
        self.stem3 = ResBlock3D(in_channels, f * 8)
        self.head0 = nn.Conv3d(f, out_channels, 1)
        self.head1 = nn.Conv3d(f * 2, out_channels, 1)
        self.head2 = nn.Conv3d(f * 4, out_channels, 1)
        self.head3 = nn.Conv3d(f * 8, out_channels, 1)
        self.allowed = {(64, 256, 256): 0, (32, 128, 128): 1, (16, 64, 64): 2, (8, 32, 32): 3}
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def infer_level(self, dhw):
        return self.allowed.get(dhw, None)

    def output_scales(self, upsample_to_input=True):
        if upsample_to_input:
            return {"logit0": "x1", "logit1": "x1", "logit2": "x1", "logit3": "x1"}
        return {"logit0": "x1", "logit1": "x1/2", "logit2": "x1/4", "logit3": "x1/8"}

    def _upsample_list(self, outs, size_3d):
        if size_3d is None:
            return outs
        return [F.interpolate(o, size=size_3d, mode='trilinear', align_corners=False) for o in outs]

    def forward(self, x, upsample_to_input=None, strict_allowed=True):
        if upsample_to_input is None:
            upsample_to_input = self.default_upsample_to_input
        N, C, D, H, W = x.shape
        L = self.infer_level((D, H, W))
        if L is None:
            if strict_allowed:
                raise ValueError(f"Input patch size {(D,H,W)} not in allowed set {list(self.allowed.keys())}")
            else:
                return []
        s0 = s1 = s2 = s3 = None
        if L == 0:
            x0 = self.enc0(x); s0 = x0
            x1 = self.enc1(self.pool0(x0)); s1 = x1
            x2 = self.enc2(self.pool1(x1)); s2 = x2
            x3 = self.enc3(self.pool2(x2)); s3 = x3
            xb = self.bottleneck(self.pool3(x3))
        elif L == 1:
            x1 = self.stem1(x); s1 = x1
            x2 = self.enc2(self.pool1(x1)); s2 = x2
            x3 = self.enc3(self.pool2(x2)); s3 = x3
            xb = self.bottleneck(self.pool3(x3))
        elif L == 2:
            x2 = self.stem2(x); s2 = x2
            x3 = self.enc3(self.pool2(x2)); s3 = x3
            xb = self.bottleneck(self.pool3(x3))
        else:
            x3 = self.stem3(x); s3 = x3
            xb = self.bottleneck(self.pool3(x3))
        outs = []
        cur = xb
        if L <= 3:
            u3 = self.up3(cur)
            s3m = _match_size_like(s3, u3)
            d3 = self.dec3(torch.cat([u3, s3m], dim=1))
            cur = d3
            logit3 = self.head3(d3)
            if L == 3:
                outs = [logit3]
        if L <= 2:
            u2 = self.up2(cur)
            s2m = _match_size_like(s2, u2)
            d2 = self.dec2(torch.cat([u2, s2m], dim=1))
            cur = d2
            logit2 = self.head2(d2)
            if L == 2:
                outs = [logit2, logit3]
        if L <= 1:
            u1 = self.up1(cur)
            s1m = _match_size_like(s1, u1)
            d1 = self.dec1(torch.cat([u1, s1m], dim=1))
            cur = d1
            logit1 = self.head1(d1)
            if L == 1:
                outs = [logit1, logit2, logit3]
        if L == 0:
            u0 = self.up0(cur)
            s0m = _match_size_like(s0, u0)
            d0 = self.dec0(torch.cat([u0, s0m], dim=1))
            logit0 = self.head0(d0)
            outs = [logit0, logit1, logit2, logit3]
        if upsample_to_input:
            outs = self._upsample_list(outs, size_3d=(D, H, W))
        return outs
