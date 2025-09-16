import torch
import torch.nn as nn
import torch.nn.functional as F

def _to_5d(logits, targets):
    if logits.dim() == 4:
        logits = logits.unsqueeze(1)
    if targets.dim() == 4:
        targets = targets.unsqueeze(1)
    return logits, targets.float()

def _prep_mask(targets, ignore_mask=None, ignore_value=None):
    N, C, D, H, W = targets.shape
    if ignore_mask is None:
        mask = targets.new_ones((N, 1, D, H, W))
    else:
        if ignore_mask.dim() == 4:
            ignore_mask = ignore_mask.unsqueeze(1)
        mask = ignore_mask.float()
    if ignore_value is not None:
        valid = (targets != ignore_value).float()
        mask = mask * valid
    return mask.clamp(0, 1)

def _masked_mean(x, mask, eps=1e-12):
    num = (x.float() * mask.float()).sum()
    den = mask.float().sum().clamp_min(eps)
    return (num / den).to(dtype=x.dtype)

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, ignore_mask=None, ignore_value=None):
        logits, targets = _to_5d(logits, targets)
        mask = _prep_mask(targets, ignore_mask, ignore_value)
        p = torch.sigmoid(logits)
        p = p * mask
        t = targets * mask
        dims = (2, 3, 4)
        inter = (p * t).sum(dim=dims).float()
        den = (p.sum(dim=dims) + t.sum(dim=dims)).float()
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        valid_fg = (t.sum(dim=dims) > 0)
        if valid_fg.any():
            return (1 - dice[valid_fg]).mean().to(dtype=logits.dtype)
        return torch.zeros([], device=logits.device, dtype=logits.dtype)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets, ignore_mask=None, ignore_value=None):
        logits, targets = _to_5d(logits, targets)
        mask = _prep_mask(targets, ignore_mask, ignore_value)
        p = torch.sigmoid(logits)
        p = p * mask
        t = targets * mask
        dims = (2, 3, 4)
        TP = (p * t).sum(dim=dims).float()
        FP = (p * (1 - t)).sum(dim=dims).float()
        FN = ((1 - p) * t).sum(dim=dims).float()
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        valid_fg = (t.sum(dim=dims) > 0)
        if valid_fg.any():
            return (1 - tversky[valid_fg]).mean().to(dtype=logits.dtype)
        return torch.zeros([], device=logits.device, dtype=logits.dtype)

class FocalLossSigmoid(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, ignore_mask=None, ignore_value=None):
        logits, targets = _to_5d(logits, targets)
        mask = _prep_mask(targets, ignore_mask, ignore_value)
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        ce_pos = -torch.log(p)
        ce_neg = -torch.log(1 - p)
        loss = targets * (self.alpha * ((1 - p) ** self.gamma) * ce_pos) + (1 - targets) * ((1 - self.alpha) * (p ** self.gamma) * ce_neg)
        if self.reduction == 'mean':
            return _masked_mean(loss, mask)
        if self.reduction == 'sum':
            return (loss.float() * mask.float()).sum().to(dtype=logits.dtype)
        return loss * mask

class ComboSingleLoss(nn.Module):
    def __init__(self, w_dice=0.3, w_focal=0.2, w_bce=0.1, w_tversky=0.4, focal_alpha=0.25, focal_gamma=2.0, tversky_alpha=0.5, tversky_beta=0.5, smooth=1e-6, pos_weight=None):
        super().__init__()
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_bce = w_bce
        self.w_tversky = w_tversky
        self.dice = SoftDiceLoss(smooth=smooth)
        self.focal = FocalLossSigmoid(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, smooth=smooth)
        self.register_buffer('_pos_weight_buf', None, persistent=False)
        if isinstance(pos_weight, torch.Tensor):
            self._pos_weight_buf = pos_weight.detach().clone()
        else:
            self._pos_weight_value = float(pos_weight) if pos_weight is not None else None

    def _bce_logits(self, logits, targets, mask, pos_w):
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w, reduction='none')
        return _masked_mean(loss, mask)

    def forward(self, logits, targets, ignore_mask=None, ignore_value=None):
        logits, targets = _to_5d(logits, targets)
        mask = _prep_mask(targets, ignore_mask, ignore_value)
        if self._pos_weight_buf is not None:
            pos_w = self._pos_weight_buf.to(device=logits.device, dtype=logits.dtype)
        elif hasattr(self, '_pos_weight_value') and self._pos_weight_value is not None:
            pos_w = torch.as_tensor(self._pos_weight_value, device=logits.device, dtype=logits.dtype)
        else:
            pos_w = None
        ld = self.dice(logits, targets, mask, ignore_value=None)
        lf = self.focal(logits, targets, mask, ignore_value=None)
        lb = self._bce_logits(logits, targets, mask, pos_w)
        lt = self.tversky(logits, targets, mask, ignore_value=None)
        return self.w_dice * ld + self.w_focal * lf + self.w_bce * lb + self.w_tversky * lt

class DeepSupervisionComboLoss(nn.Module):
    def __init__(self, head_weights=(0.5, 0.3, 0.15, 0.05), w_dice=0.3, w_focal=0.2, w_bce=0.1, w_tversky=0.4, focal_alpha=0.25, focal_gamma=2.0, tversky_alpha=0.5, tversky_beta=0.5, smooth=1e-6, pos_weight=None):
        super().__init__()
        self.head_weights = tuple(head_weights)
        self.single = ComboSingleLoss(w_dice=w_dice, w_focal=w_focal, w_bce=w_bce, w_tversky=w_tversky, focal_alpha=focal_alpha, focal_gamma=focal_gamma, tversky_alpha=tversky_alpha, tversky_beta=tversky_beta, smooth=smooth, pos_weight=pos_weight)

    def forward(self, outputs_list, targets, ignore_mask=None, ignore_value=None):
        if isinstance(outputs_list, torch.Tensor):
            outputs_list = [outputs_list]
        if targets.dim() == 4:
            targets = targets.unsqueeze(1).float()
        else:
            targets = targets.float()
        if ignore_mask is not None and ignore_mask.dim() == 4:
            ignore_mask = ignore_mask.unsqueeze(1)
        D, H, W = targets.shape[-3:]
        used = len(outputs_list)
        w = torch.as_tensor(self.head_weights[:used], device=targets.device, dtype=targets.dtype)
        w = w / (w.sum() + 1e-12)
        total = torch.zeros([], device=targets.device, dtype=targets.dtype)
        for i, logits in enumerate(outputs_list):
            if logits.shape[-3:] != (D, H, W):
                logits = F.interpolate(logits, size=(D, H, W), mode='trilinear', align_corners=False)
            if ignore_mask is not None and ignore_mask.shape[-3:] != (D, H, W):
                m = F.interpolate(ignore_mask.float(), size=(D, H, W), mode='nearest')
            else:
                m = ignore_mask
            total = total + w[i] * self.single(logits, targets, ignore_mask=m, ignore_value=ignore_value)
        return total
