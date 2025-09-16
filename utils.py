import os
import json
import logging
import time
import subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _fmt6(x):
    try:
        v = float(x)
        if np.isnan(v):
            return "NaN"
        return f"{v:.6f}"
    except Exception:
        return "None"


def _round6(x):
    try:
        return round(float(x), 6)
    except Exception:
        return None


def setup_logging(checkpoint_dir, filename='train.log', level=logging.INFO):
    ensure_dir(checkpoint_dir)
    log_file = os.path.join(checkpoint_dir, filename)
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f'Logging to {log_file}')


def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def get_unique_dir(base_path):
    ts_path = f"{base_path}_{_timestamp()}"
    if not os.path.exists(ts_path):
        return ts_path
    version = 2
    while True:
        new_path = f"{ts_path}_v{version}"
        if not os.path.exists(new_path):
            return new_path
        version += 1


def get_timestamped_dir(base_path, suffix=None):
    ts = _timestamp()
    p = f"{base_path}_{ts}" if suffix is None else f"{base_path}_{ts}_{suffix}"
    if not os.path.exists(p):
        return p
    version = 2
    while True:
        q = f"{p}_v{version}"
        if not os.path.exists(q):
            return q
        version += 1


def atomic_torch_save(state, path):
    ensure_dir(os.path.dirname(path))
    tmp = f"{path}.tmp_{os.getpid()}"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _git_commit_short():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def current_lr(optimizer):
    try:
        lrs = [g["lr"] for g in optimizer.param_groups]
        if len(lrs) == 0:
            return None
        if all(abs(l - lrs[0]) < 1e-12 for l in lrs):
            return lrs[0]
        return lrs
    except Exception:
        return None


def build_checkpoint_state(model,
                           optimizer=None,
                           scheduler=None,
                           scaler=None,
                           epoch=0,
                           train_loss=None,
                           val_loss=None,
                           best_val=None,
                           meta=None):
    state = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val": best_val,
        "meta": {
            "lr": current_lr(optimizer) if optimizer is not None else None,
            "git_commit": _git_commit_short(),
        }
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if isinstance(meta, dict):
        state["meta"].update(meta)
    return state


def save_checkpoint(state, checkpoint_dir, is_best=False, filename_prefix='ckpt', periodic_every: int = 50):
    ensure_dir(checkpoint_dir)
    last_path = os.path.join(checkpoint_dir, f'{filename_prefix}_last.pt')
    atomic_torch_save(state, last_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, f'{filename_prefix}_best.pt')
        atomic_torch_save(state, best_path)

    ep = state.get('epoch', None)
    is_periodic = False
    if periodic_every and isinstance(ep, int) and ep is not None and ep >= 0:
        is_periodic = ((ep + 1) % periodic_every == 0)
        if is_periodic:
            ep_path = os.path.join(checkpoint_dir, f'{filename_prefix}_ep{ep}.pt')
            atomic_torch_save(state, ep_path)

    tl = state.get('train_loss', None)
    vl = state.get('val_loss', None)
    vd = state.get('best_val', None)

    logging.info(
        f"Checkpoint saved: epoch={ep}, train_loss={_fmt6(tl)}, val_loss={_fmt6(vl)}, "
        f"best_val={_fmt6(vd)}, best={is_best}, periodic={'yes' if is_periodic else 'no'}"
    )


def load_checkpoint_path(model, optimizer, scheduler, scaler, ckpt_path, device, strict_model=False):
    if ckpt_path is None or ckpt_path == "" or not os.path.isfile(ckpt_path):
        return model, optimizer, scheduler, scaler, 0, -1.0, None, None, False
    try:
        ckpt = torch.load(ckpt_path, map_location=device)

        did_resume = False
        if "model" in ckpt:
            res = model.load_state_dict(ckpt["model"], strict=strict_model)
            if not strict_model and hasattr(res, "missing_keys"):
                mk = res.missing_keys
                uk = res.unexpected_keys
                if mk:
                    logging.warning(f"Model missing keys: {mk}")
                if uk:
                    logging.warning(f"Model unexpected keys: {uk}")
            did_resume = True
        else:
            logging.warning("Checkpoint missing 'model' key.")

        if optimizer is not None and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                logging.warning(f"Failed to load optimizer state: {e}")
        if scheduler is not None and "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                logging.warning(f"Failed to load scheduler state: {e}")
        if scaler is not None and "scaler" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                logging.warning(f"Failed to load scaler state: {e}")

        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val", -1.0)
        train_loss = ckpt.get("train_loss", None)
        val_loss = ckpt.get("val_loss", None)
        meta = ckpt.get("meta", {})
        logging.info(
            f"Loaded checkpoint from {ckpt_path} "
            f"(epoch={start_epoch}, best_val={_fmt6(best_val)}, lr={meta.get('lr', None)}, git={meta.get('git_commit', None)})"
        )
        return model, optimizer, scheduler, scaler, start_epoch, best_val, train_loss, val_loss, did_resume
    except (FileNotFoundError, KeyError, RuntimeError, EOFError) as e:
        logging.error(f"Error loading checkpoint from {ckpt_path}: {e}")
        return model, optimizer, scheduler, scaler, 0, -1.0, None, None, False


def save_checkpoint_legacy(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir, best_dice_score, filename_prefix="checkpoint"):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val": best_dice_score,
        "meta": {
            "git_commit": _git_commit_short()
        }
    }
    save_checkpoint(state, checkpoint_dir, is_best=False, filename_prefix=filename_prefix)


def load_checkpoint_legacy(model, optimizer, checkpoint_dir, filename="checkpoint.pth", device="cpu", strict_model=False):
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(path):
        logging.warning(f"No checkpoint at {path}")
        return 1, None, None, 0.0, False
    try:
        ckpt = torch.load(path, map_location=device)
        res = model.load_state_dict(ckpt["model"], strict=strict_model)
        if not strict_model and hasattr(res, "missing_keys"):
            mk = res.missing_keys
            uk = res.unexpected_keys
            if mk:
                logging.warning(f"Model missing keys: {mk}")
            if uk:
                logging.warning(f"Model unexpected keys: {uk}")
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        epoch = ckpt.get("epoch", 1)
        train_loss = ckpt.get("train_loss", None)
        val_loss = ckpt.get("val_loss", None)
        best_dice_score = ckpt.get("best_val", 0.0)
        logging.info(f"Checkpoint loaded from {path} (epoch={epoch}, best_val={_fmt6(best_dice_score)})")
        return epoch, train_loss, val_loss, best_dice_score, True
    except (FileNotFoundError, KeyError, RuntimeError, EOFError) as e:
        logging.error(f"Error loading checkpoint from {path}: {e}")
        return 1, None, None, 0.0, False


def _apply_ignore_mask(t, ignore_mask=None, ignore_value=None):
    if ignore_mask is not None:
        if isinstance(ignore_mask, torch.Tensor):
            m = ignore_mask
        else:
            m = torch.as_tensor(ignore_mask, dtype=torch.float32, device=t.device)
        if m.dim() == t.dim() - 1:
            m = m.unsqueeze(0)
        if m.dim() == 4 and t.dim() == 5:
            m = m.unsqueeze(1)
        m = (m > 0).to(dtype=t.dtype)
        t = t * m
    if ignore_value is not None:
        iv = torch.as_tensor(ignore_value, dtype=t.dtype, device=t.device)
        t = torch.where(torch.isclose(t, iv), torch.zeros_like(t), t)
    return t


def calculate_dice_score(preds,
                         targets,
                         eps=1e-6,
                         threshold=0.5,
                         from_logits=False,
                         ignore_mask=None,
                         ignore_value=None,
                         empty_as_one=False):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach()

    preds = torch.as_tensor(preds)
    targets = torch.as_tensor(targets)

    if preds.dim() == 4:
        preds = preds.unsqueeze(1)
    if targets.dim() == 4:
        targets = targets.unsqueeze(1)

    if preds.dtype.is_floating_point:
        if from_logits:
            preds = torch.sigmoid(preds)
        preds = (preds >= threshold).to(torch.float32)
    else:
        preds = (preds == 1).to(torch.float32)

    if targets.dtype.is_floating_point:
        targets = (targets >= 0.5).to(torch.float32)
    else:
        targets = (targets == 1).to(torch.float32)

    preds = _apply_ignore_mask(preds, ignore_mask, ignore_value)
    targets = _apply_ignore_mask(targets, ignore_mask, ignore_value)

    intersection = (preds * targets).sum().float()
    pred_sum = preds.sum().float()
    target_sum = targets.sum().float()
    union = pred_sum + target_sum

    if target_sum.item() == 0.0 and pred_sum.item() == 0.0:
        if empty_as_one:
            return 1.0, 0, 0, 0, 0
        else:
            return None, 0, 0, 0, 0

    dice = ((2.0 * intersection + eps) / (union + eps)).item()
    return dice, int(target_sum.item()), int(pred_sum.item()), int(intersection.item()), int(union.item())


def load_loss_history(checkpoint_dir):
    history_file = os.path.join(checkpoint_dir, "loss_history.json")
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        return history.get("train_losses", []), history.get("val_losses", []), history.get("val_dices", [])
    else:
        return [], [], []


def _to_float_list_safe(xs):
    out = []
    for x in xs:
        if x is None:
            out.append(float('nan'))
        else:
            try:
                out.append(float(x))
            except Exception:
                out.append(float('nan'))
    return out


def save_loss_history(train_losses, val_losses, val_dices, checkpoint_dir):
    ensure_dir(checkpoint_dir)
    history = {
        "train_losses": [_round6(v) for v in train_losses],
        "val_losses": [_round6(v) for v in val_losses],
        "val_dices": [_round6(v) for v in val_dices],
    }
    with open(os.path.join(checkpoint_dir, "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _nan_to_none(xs):
    out = []
    for x in xs:
        if x is None:
            out.append(None)
        else:
            try:
                v = float(x)
                out.append(None if np.isnan(v) else v)
            except Exception:
                out.append(None)
    return out


def save_loss_graph(train_losses, val_losses, val_dices, checkpoint_dir):
    ensure_dir(checkpoint_dir)
    tr = _nan_to_none(train_losses)
    va = _nan_to_none(val_losses)
    vd = _nan_to_none(val_dices)

    plt.figure(figsize=(10, 6))
    if any(v is not None for v in tr):
        plt.plot([v if v is not None else np.nan for v in tr], label='Train Loss')
    if any(v is not None for v in va):
        plt.plot([v if v is not None else np.nan for v in va], label='Validation Loss')
    if any(v is not None for v in vd):
        plt.plot([v if v is not None else np.nan for v in vd], label='Validation Dice')
    plt.xlabel('Epoch')
    plt.title('Loss & Dice over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "loss_curve.png"))
    plt.close()
