import math
import logging
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import calculate_dice_score


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        max_grad_norm: float = 0.0,
        accum_steps: int = 1,
        amp_enabled: bool = True,
        logger: Optional[logging.Logger] = None,
        scheduler_step_on: Optional[str] = None,
        log_interval: int = 50,
        ignore_value: Optional[float] = None,
        ignore_mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]] = None,
        empty_as_one: bool = False,
        *,
        ema: Optional[object] = None,
        use_channels_last: bool = False,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler or torch.cuda.amp.GradScaler()
        self.max_grad_norm = float(max(0.0, max_grad_norm))
        self.accum_steps = max(1, int(accum_steps))
        self.amp_enabled = bool(amp_enabled)
        self.logger = logger or logging.getLogger(__name__)
        self.scheduler_step_on = scheduler_step_on
        self.log_interval = max(1, int(log_interval))
        self.ignore_value = ignore_value
        self.ignore_mask_fn = ignore_mask_fn
        self.empty_as_one = bool(empty_as_one)
        # new:
        self.ema = ema
        self.use_channels_last = bool(use_channels_last)

    @staticmethod
    def _finite_stats(t: torch.Tensor) -> Tuple[float, float, float]:
        mn = float(t.min().item())
        mx = float(t.max().item())
        fr = float(torch.mean(torch.isfinite(t).float()).item())
        return mn, mx, fr

    def _check_logits_finite(self, outs: Union[List[torch.Tensor], torch.Tensor]) -> bool:
        if isinstance(outs, list):
            all_ok = True
            for li, o in enumerate(outs):
                if not torch.isfinite(o).all():
                    mn, mx, fr = self._finite_stats(o)
                    self.logger.warning(f"[NaN logits] head={li} finite_ratio={fr:.3f} min={mn:.3e} max={mx:.3e} shape={tuple(o.shape)}")
                    all_ok = False
            return all_ok
        elif isinstance(outs, torch.Tensor):
            if not torch.isfinite(outs).all():
                mn, mx, fr = self._finite_stats(outs)
                self.logger.warning(f"[NaN logits] finite_ratio={fr:.3f} min={mn:.3e} max={mx:.3e} shape={tuple(outs.shape)}")
                return False
            return True
        else:
            return True

    def train_one_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        used_batches = 0
        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc="Train", ncols=110, leave=False)
        step_since_log = 0
        pending_steps = 0

        for step, (img, lbl) in enumerate(pbar, start=1):
            if not torch.isfinite(img).all():
                self.logger.warning("Input has non-finite values; skipping batch")
                continue
            img = img.to(self.device, non_blocking=True)
            if self.use_channels_last:
                img = img.to(memory_format=torch.channels_last)
            lbl = lbl.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                outs = self.model(img)
            if isinstance(outs, list) and len(outs) == 0:
                pbar.set_postfix(loss="skip")
                continue
            if not self._check_logits_finite(outs):
                continue

            mask = None
            if self.ignore_mask_fn is not None:
                try:
                    mask = self.ignore_mask_fn(img, lbl)
                except Exception:
                    mask = None
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                loss = self.criterion(outs, lbl, ignore_mask=mask, ignore_value=self.ignore_value)

            if isinstance(loss, torch.Tensor):
                if not torch.isfinite(loss).item():
                    self.logger.warning("Non-finite loss; skipping batch")
                    continue
            else:
                try:
                    f = float(loss)
                    if not math.isfinite(f):
                        self.logger.warning("Non-finite loss(float); skipping batch")
                        continue
                    loss = torch.as_tensor(f, device=self.device)
                except Exception:
                    self.logger.warning("Invalid loss; skipping batch")
                    continue

            loss_to_backprop = loss / self.accum_steps
            self.scaler.scale(loss_to_backprop).backward()

            if self.max_grad_norm > 0.0 and (step % self.accum_steps) == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if (step % self.accum_steps) == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                pending_steps = 0
                # EMA update after optimizer step
                if self.ema is not None:
                    self.ema.update(self.model)
                if self.scheduler is not None and self.scheduler_step_on == "iter":
                    self.scheduler.step()
            else:
                pending_steps += 1

            total_loss += float(loss.detach().item())
            used_batches += 1
            step_since_log += 1
            if step_since_log >= self.log_interval:
                lr = self.optimizer.param_groups[0]["lr"] if len(self.optimizer.param_groups) > 0 else None
                pbar.set_postfix(loss=f"{(total_loss/used_batches):.6f}", lr=(f"{lr:.6f}" if lr is not None else "n/a"))
                step_since_log = 0

        if pending_steps > 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.model)
            if self.scheduler is not None and self.scheduler_step_on == "iter":
                self.scheduler.step()

        if self.scheduler is not None and self.scheduler_step_on == "epoch":
            self.scheduler.step()

        if used_batches == 0:
            self.logger.warning("[train] all batches were skipped.")
            return 0.0
        return total_loss / used_batches

    @torch.no_grad()
    def validate(self, loader) -> Tuple[Optional[float], float]:
        self.model.eval()
        dices: List[float] = []
        total_val_loss = 0.0
        used_batches = 0
        badloss_batches = 0
        skipped_batches = 0
        pbar = tqdm(loader, desc="Val", ncols=110, leave=False)
        step_since_log = 0

        for img, lbl in pbar:
            img = img.to(self.device, non_blocking=True)
            if self.use_channels_last:
                img = img.to(memory_format=torch.channels_last)
            lbl = lbl.to(self.device, non_blocking=True)
            outs = self.model(img)
            if isinstance(outs, list) and len(outs) == 0:
                skipped_batches += 1
                cur_d = (sum(dices) / len(dices)) if dices else 0.0
                pbar.set_postfix(loss="skip", dice=f"{cur_d:.6f}")
                continue
            if not self._check_logits_finite(outs):
                skipped_batches += 1
                cur_d = (sum(dices) / len(dices)) if dices else 0.0
                pbar.set_postfix(loss="nan/inf", dice=f"{cur_d:.6f}")
                continue

            cur_loss = None
            mask = None
            if self.ignore_mask_fn is not None:
                try:
                    mask = self.ignore_mask_fn(img, lbl)
                except Exception:
                    mask = None

            if self.criterion is not None:
                val_loss = self.criterion(outs, lbl, ignore_mask=mask, ignore_value=self.ignore_value)
                bad_loss = False
                if isinstance(val_loss, torch.Tensor):
                    if not torch.isfinite(val_loss).item():
                        bad_loss = True
                else:
                    try:
                        f = float(val_loss)
                        if not math.isfinite(f):
                            bad_loss = True
                        else:
                            val_loss = f
                    except Exception:
                        bad_loss = True
                if bad_loss:
                    badloss_batches += 1
                    cur_d = (sum(dices) / len(dices)) if dices else 0.0
                    pbar.set_postfix(loss="nan/inf", dice=f"{cur_d:.6f}")
                    continue
                total_val_loss += float(val_loss)
                cur_loss = float(val_loss)

            logits = outs[0] if isinstance(outs, list) else outs
            if logits.shape[-3:] != lbl.shape[-3:]:
                logits = F.interpolate(logits, size=lbl.shape[-3:], mode='trilinear', align_corners=False)
            N = lbl.shape[0]
            if mask is not None and mask.dim() == 4 and mask.shape[0] == N:
                pass
            elif mask is not None and mask.dim() == 5 and mask.shape[0] == N:
                mask = mask.squeeze(1)
            else:
                mask = None

            for n in range(N):
                d, _, _, _, _ = calculate_dice_score(
                    logits[n:n+1],
                    lbl[n],
                    eps=1e-6,
                    threshold=0.5,
                    from_logits=True,
                    ignore_mask=(mask[n] if mask is not None else None),
                    ignore_value=self.ignore_value,
                    empty_as_one=self.empty_as_one
                )
                if d is not None:
                    dices.append(float(d))
            used_batches += 1
            step_since_log += 1
            mean_d = (sum(dices) / len(dices)) if dices else 0.0
            if step_since_log >= self.log_interval:
                pbar.set_postfix(loss=(f"{cur_loss:.6f}" if cur_loss is not None else "n/a"), dice=f"{mean_d:.6f}")
                step_since_log = 0

        mean_val_loss = (total_val_loss / used_batches) if (self.criterion is not None and used_batches > 0) else None
        mean_val_dice = float(sum(dices) / len(dices)) if len(dices) > 0 else 0.0
        try:
            self.logger.info(f"[val] used={used_batches} skipped={skipped_batches} badloss={badloss_batches}")
        except Exception:
            pass
        return mean_val_loss, mean_val_dice
