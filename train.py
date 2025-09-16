import os
import argparse
import json
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from model import MultiScaleRoutedResUNet3D_DS
from dataloader import (
    StageAwareMedicalDataset3D,
    make_stage_dataloader_single,
    make_stage_dataloaders_multi,
    normalize_ct,  # for SW inference
)
from loss import DeepSupervisionComboLoss
from utils import (
    setup_logging,
    ensure_dir,
    get_unique_dir,
    save_checkpoint,
    load_checkpoint_path,
    load_loss_history,
    save_loss_history,
    save_loss_graph,
    calculate_dice_score,
)
from trainer import Trainer
from concurrent.futures import ThreadPoolExecutor, as_completed
import SimpleITK as sitk


# ---------------------- helpers ----------------------
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    import numpy as np, random, torch
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "set_seed"):
        info.dataset.set_seed(int(s))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def replace_instancenorm_with_groupnorm(model, num_groups=8):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.InstanceNorm3d):
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name != "" else model
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=module.num_features, affine=True)
            setattr(parent, attr_name, gn)
    return model


def _read_shape(path):
    p = path.lower()
    if p.endswith(".npy"):
        a = np.load(path, mmap_mode="r")
        return tuple(a.shape)
    if p.endswith(".npz"):
        with np.load(path) as z:
            return tuple(z["arr"].shape)
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return tuple(arr.shape)


def _can_fit(shape, size):
    D, H, W = shape
    sd, sh, sw = size
    return (sd <= D) and (sh <= H) and (sw <= W)


def _case_to_paths(case: str, base_dir: str):
    prefix = case.split("_")[0]
    rest = "_".join(case.split("_")[1:])
    core = rest.rsplit("_", 1)[0]

    def choose(root, base_noext):
        npy = os.path.join(root, base_noext + ".npy")
        if os.path.exists(npy):
            return npy
        npz = os.path.join(root, base_noext + ".npz")
        if os.path.exists(npz):
            return npz
        nii = os.path.join(root, base_noext + ".nii.gz")
        return nii

    img_path = choose(os.path.join(base_dir, prefix, "img_npy"), core)
    lbl_path = choose(os.path.join(base_dir, prefix, "lbl_npy"), core.replace("_img", "_lbl"))
    return img_path, lbl_path


def filter_cases_for_sizes_mp(img_paths, lbl_paths, sizes, workers: int = 16):
    assert len(img_paths) == len(lbl_paths)
    total = len(img_paths)
    kept_indices = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_read_shape, lp): i for i, lp in enumerate(lbl_paths)}
        for fut in as_completed(futs):
            i = futs[fut]
            shp = fut.result()
            if any(_can_fit(shp, ps) for ps in sizes):
                kept_indices.append(i)
    kept_indices.sort()
    fimg = [img_paths[i] for i in kept_indices]
    flbl = [lbl_paths[i] for i in kept_indices]
    return fimg, flbl, len(kept_indices), total


def load_cases_mp(txt_path: str, base_dir: str, workers: int = 16, check_exists: bool = True):
    with open(txt_path, "r", encoding="utf-8") as f:
        cases = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    img_paths, lbl_paths, missing = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_case_to_paths, c, base_dir): c for c in cases}
        for fut in as_completed(futures):
            img_path, lbl_path = fut.result()
            if check_exists and not (os.path.exists(img_path) and os.path.exists(lbl_path)):
                missing.append((img_path, lbl_path))
            img_paths.append(img_path)
            lbl_paths.append(lbl_path)
    if check_exists and missing:
        for (ip, lp) in missing[:10]:
            logging.warning(f"Missing file(s): img={ip} | lbl={lp}")
        logging.warning(f"Total missing pairs: {len(missing)}")
    return img_paths, lbl_paths


def build_stage_scheduler(
    optimizer,
    sched_name: str,
    num_epochs: int,
    warmup_epochs: int = 0,
    eta_min: float = 1e-6,
    last_epoch: int = -1,
):
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        CosineAnnealingWarmRestarts,
        SequentialLR,
        LinearLR,
        ReduceLROnPlateau,
        LambdaLR,
    )

    if sched_name == "cos":
        if warmup_epochs > 0 and warmup_epochs < num_epochs:
            warm = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            main = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=eta_min,
                last_epoch=last_epoch if last_epoch >= 0 else -1,
            )
            return SequentialLR(optimizer, schedulers=[warm, main], milestones=[warmup_epochs], last_epoch=last_epoch)
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min, last_epoch=last_epoch)

    if sched_name == "coswr":
        return CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs, T_mult=1, eta_min=eta_min, last_epoch=last_epoch)

    if sched_name == "poly":
        power = 0.9
        return LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / max(1, num_epochs)) ** power, last_epoch=last_epoch)

    if sched_name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, verbose=False, min_lr=eta_min)

    raise ValueError(f"Unknown scheduler: {sched_name}")


def subsample_by_prefix(img_paths, lbl_paths, base_dir, quotas, seed=123):
    assert len(img_paths) == len(lbl_paths)
    buckets = {k: [] for k in quotas.keys()}
    for i, p in enumerate(img_paths):
        rel = os.path.relpath(p, base_dir)
        prefix = rel.split(os.sep)[0]
        if prefix in buckets:
            buckets[prefix].append(i)
    rng = random.Random(seed)
    keep_indices = []
    for prefix, idxs in buckets.items():
        rng.shuffle(idxs)
        keep_indices.extend(idxs[: quotas[prefix]] if prefix in quotas else [])
    keep_indices.sort()
    img_keep = [img_paths[i] for i in keep_indices]
    lbl_keep = [lbl_paths[i] for i in keep_indices]
    return img_keep, lbl_keep


def _resolve_resume_path(resume: str, ckpt_dir: str) -> str:
    if not resume:
        return ""
    if resume in ("auto", "last"):
        return os.path.join(ckpt_dir, "ckpt_last.pt")
    if os.path.isdir(resume):
        candidate = os.path.join(resume, "ckpt_last.pt")
        return candidate if os.path.exists(candidate) else ""
    return resume


def _stage_idx_and_offset(global_ep: int, stages, stage_epochs):
    cum = 0
    for i, st in enumerate(stages):
        L = stage_epochs[st["name"]]
        if global_ep < cum + L:
            return i, global_ep - cum
        cum += L
    return len(stages) - 1, stage_epochs[stages[-1]["name"]]


# ---------------------- EMA ----------------------
class EMA:
    """Simple EMA of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.decay = decay
        self.shadow = {}
        self.device = device
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()
                if device is not None:
                    self.shadow[name] = self.shadow[name].to(device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            new = p.detach()
            if self.device is not None:
                new = new.to(self.device)
            self.shadow[name].mul_(self.decay).add_(new, alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        """Swap current params with EMA params (in-place). Returns a backup dict."""
        backup = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                backup[name] = p.detach().clone()
                p.copy_(self.shadow[name].to(p.device))
        return backup

    def restore(self, model: nn.Module, backup):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                p.copy_(backup[name].to(p.device))


# ---------------------- Sliding-window inference ----------------------
@torch.no_grad()
def sliding_window_validate_cases(
    model: nn.Module,
    img_paths,
    lbl_paths,
    patch_size,          # (D,H,W)
    overlap=0.5,
    device="cuda",
    use_channels_last=True,
    use_ema: Optional[EMA] = None,
):
    """
    Full-volume validation with a simple tiling & averaging scheme.
    Returns mean Dice over provided cases (ignores empty-empty cases).
    """
    assert len(img_paths) == len(lbl_paths)
    model.eval()
    if use_ema is not None:
        backup = use_ema.apply_to(model)
    dices = []

    for ip, lp in zip(img_paths, lbl_paths):
        # load full volumes
        vol = _load_3d(ip)
        gt  = _load_3d(lp)
        gt = (gt > 0).astype(np.float32)
        vol = normalize_ct(vol.astype(np.float32, copy=False))

        # tile infer
        logits = _sliding_infer_3d(model, vol, patch_size, overlap, device, use_channels_last)
        # dice
        d, _, _, _, _ = calculate_dice_score(
            torch.from_numpy(logits)[None, None],  # [1,1,D,H,W]
            torch.from_numpy(gt)[None],           # [1,D,H,W]
            from_logits=True,
            threshold=0.5,
            empty_as_one=False
        )
        if d is not None:
            dices.append(float(d))

    if use_ema is not None:
        use_ema.restore(model, backup)

    return float(np.mean(dices)) if dices else 0.0


def _load_3d(path: str):
    p = path.lower()
    if p.endswith(".npy"):
        return np.load(path, mmap_mode="r").astype(np.float32)
    if p.endswith(".npz"):
        with np.load(path) as z:
            return z["arr"].astype(np.float32)
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(np.float32)


def _sliding_infer_3d(model, vol_np, patch_size, overlap, device, use_channels_last):
    """Return raw logits volume (numpy, float32) with same shape as input."""
    model = model.to(device)
    D, H, W = vol_np.shape
    sd, sh, sw = patch_size
    # stride
    sz = max(1, int(sd * (1 - overlap)))
    sy = max(1, int(sh * (1 - overlap)))
    sx = max(1, int(sw * (1 - overlap)))
    # output buffers
    out_sum = np.zeros((D, H, W), dtype=np.float32)
    out_cnt = np.zeros((D, H, W), dtype=np.float32)

    # iterate
    for z0 in range(0, max(1, D - sd + 1), sz):
        for y0 in range(0, max(1, H - sh + 1), sy):
            for x0 in range(0, max(1, W - sw + 1), sx):
                z1, y1, x1 = z0 + sd, y0 + sh, x0 + sw
                zp = min(z1, D); yp = min(y1, H); xp = min(x1, W)
                # crop with simple edge handling (pad if needed)
                patch = np.zeros((sd, sh, sw), dtype=np.float32)
                cz1, cy1, cx1 = zp - z0, yp - y0, xp - x0
                patch[:cz1, :cy1, :cx1] = vol_np[z0:zp, y0:yp, x0:xp]
                ten = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
                if use_channels_last:
                    ten = ten.to(memory_format=torch.channels_last)
                with torch.cuda.amp.autocast(enabled=False):  # keep eval stable
                    outs = model(ten, upsample_to_input=True)
                    logit = outs[0] if isinstance(outs, list) else outs
                logit = logit.squeeze(0).squeeze(0).float().detach().cpu().numpy()  # [sd,sh,sw]
                out_sum[z0:zp, y0:yp, x0:xp] += logit[:cz1, :cy1, :cx1]
                out_cnt[z0:zp, y0:yp, x0:xp] += 1.0

    out_cnt[out_cnt == 0] = 1.0
    return (out_sum / out_cnt).astype(np.float32)


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--unique_dir", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--start_epoch", type=int, default=-1)
    ap.add_argument("--use_groupnorm", action="store_true")
    ap.add_argument("--gn_groups", type=int, default=16)
    ap.add_argument("--max_grad_norm", type=float, default=0.0)
    ap.add_argument("--pos_weight", type=float, default=None)
    ap.add_argument("--accum_steps", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--train_workers", type=int, default=32)
    ap.add_argument("--val_workers", type=int, default=0)
    ap.add_argument("--fast_cudnn", action="store_true")
    ap.add_argument("--sched", type=str, default="cos", choices=["cos", "coswr", "poly", "plateau"])
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--eta_min", type=float, default=1e-6)
    ap.add_argument("--plateau_metric", type=str, default="val_loss", choices=["val_loss", "val_dice"])
    ap.add_argument("--plateau_patience", type=int, default=8)
    ap.add_argument("--plateau_factor", type=float, default=0.5)

    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--channels_last", action="store_true", help="Use channels_last memory format")

    args = ap.parse_args()

    seed_everything(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = "/workspace/0_Project/curation/New/data_v2/internal"
    ckpt_dir = "/workspace/0_Project/curation/New/checkpoints/Multipatch3"
    train_txt = os.path.join(base_dir, "split_case/1_train_cases.txt")
    val_txt = os.path.join(base_dir, "split_case/2_val_cases.txt")

    if args.unique_dir:
        ckpt_dir = get_unique_dir(ckpt_dir)
    ensure_dir(ckpt_dir)
    setup_logging(ckpt_dir)

    # (1) save config snapshot
    CONFIG = {
        "args": vars(args),
        "constants": {
            "HEAD_WEIGHTS": (0.5, 0.3, 0.15, 0.05),
        },
        "paths": {"base_dir": base_dir, "ckpt_dir": ckpt_dir},
    }
    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    train_imgs, train_lbls = load_cases_mp(train_txt, base_dir, workers=16, check_exists=True)
    val_imgs, val_lbls = load_cases_mp(val_txt, base_dir, workers=16, check_exists=True)
    train_imgs, train_lbls = subsample_by_prefix(
        train_imgs, train_lbls, base_dir, quotas={"ASAN": 200, "BSNUH": 200}, seed=123
    )
    logging.info(f"Train cases: {len(train_imgs)} | Val cases: {len(val_imgs)}")

    model = MultiScaleRoutedResUNet3D_DS(in_channels=1, out_channels=1, base_features=32)
    if args.use_groupnorm:
        model = replace_instancenorm_with_groupnorm(model, num_groups=args.gn_groups)
    model = model.to(device)
    # (5) channels-last for speed (optional)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    n_params = count_parameters(model)
    logging.info(f"Total trainable parameters: {n_params}")

    HEAD_WEIGHTS = (0.5, 0.3, 0.15, 0.05)
    pos_w = torch.tensor(args.pos_weight, device=device) if args.pos_weight is not None else None
    criterion = DeepSupervisionComboLoss(
        head_weights=HEAD_WEIGHTS,
        w_dice=0.3, w_focal=0.2, w_bce=0.1, w_tversky=0.4,
        focal_alpha=0.25, focal_gamma=2.0,
        tversky_alpha=0.5, tversky_beta=0.5,
        pos_weight=pos_w,
    )

    lr = 1e-3
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ALL_PS = [(64, 256, 256), (32, 128, 128), (16, 64, 64), (8, 32, 32)]
    stages = [
        {"name": "StageA_L0", "patch_sizes": [(64, 256, 256)]},
        {"name": "StageB_L1", "patch_sizes": [(32, 128, 128)]},
        {"name": "StageC_L2", "patch_sizes": [(16, 64, 64)]},
        {"name": "StageD_L3", "patch_sizes": [(8, 32, 32)]},
        {"name": "StageE_JointFT", "patch_sizes": ALL_PS},
    ]
    stage_epochs = {
        "StageA_L0": 50,
        "StageB_L1": 50,
        "StageC_L2": 50,
        "StageD_L3": 50,
        "StageE_JointFT": 100,
    }

    STAGE2BS = {"StageA_L0": 2, "StageB_L1": 4, "StageC_L2": 8, "StageD_L3": 16, "StageE_JointFT": 1}
    STAGE2ACC = {"StageA_L0": 4, "StageB_L1": 4, "StageC_L2": 2, "StageD_L3": 1, "StageE_JointFT": 4}

    scaler = torch.cuda.amp.GradScaler()

    resume_path = _resolve_resume_path(args.resume, ckpt_dir)
    (model, optimizer, _scheduler_ignored, scaler, start_epoch, best_val, _tr_loss, _vl_loss, did_resume) = \
        load_checkpoint_path(model, optimizer, scheduler=None, scaler=scaler, ckpt_path=resume_path, device=device)
    logging.info("Resumed from checkpoint." if did_resume else "No valid resume checkpoint. Starting fresh.")
    if args.start_epoch >= 0:
        start_epoch = args.start_epoch

    train_losses, val_losses, val_dices = load_loss_history(ckpt_dir)
    start_stage, offset_in_stage = _stage_idx_and_offset(start_epoch, stages, stage_epochs)
    start_epoch_global = sum(stage_epochs[s["name"]] for s in stages[:start_stage])

    # (6) EMA
    ema = EMA(model, decay=args.ema_decay, device=device)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,  # will be set per stage
        device=device,
        scaler=scaler,
        max_grad_norm=args.max_grad_norm,
        accum_steps=args.accum_steps,  # overridden per stage
        logger=logging.getLogger(),
        amp_enabled=(not args.no_amp),
        ignore_value=None,
        ignore_mask_fn=None,
        empty_as_one=False,
        ema=ema,  # pass EMA to trainer for updates
        use_channels_last=bool(args.channels_last),
    )

    # cache full val lists once
    full_val_imgs_all, full_val_lbls_all = val_imgs, val_lbls

    for si, st in enumerate(stages):
        if si < start_stage:
            continue
        st_name = st["name"]; st_ps = st["patch_sizes"]
        num_epochs = stage_epochs[st_name]
        start_k = offset_in_stage + 1 if si == start_stage and start_epoch > 0 else 1
        offset_in_stage = 0

        logging.info(f"==> START {st_name} (resume from k={start_k}/{num_epochs})")

        scheduler = build_stage_scheduler(
            optimizer, sched_name=args.sched, num_epochs=num_epochs,
            warmup_epochs=args.warmup_epochs, eta_min=args.eta_min, last_epoch=-1
        )
        if args.sched == "plateau":
            trainer.scheduler = None
            trainer.scheduler_step_on = None
        else:
            trainer.scheduler = scheduler
            trainer.scheduler_step_on = "epoch"

        # filter cases that fit current stage sizes
        st_train_imgs, st_train_lbls, kept_tr, tot_tr = filter_cases_for_sizes_mp(train_imgs, train_lbls, st_ps, workers=16)
        st_val_imgs,   st_val_lbls,   kept_va, tot_va = filter_cases_for_sizes_mp(val_imgs,   val_lbls,   st_ps, workers=16)
        logging.info(f"[{st_name}] train kept {kept_tr}/{tot_tr} | val kept {kept_va}/{tot_va}")
        if kept_tr == 0 or kept_va == 0:
            logging.warning(f"[{st_name}] no cases fit sizes {st_ps}; skipping this stage.")
            continue

        if args.fast_cudnn:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        nw_train = args.train_workers
        nw_val   = args.val_workers
        bs_this  = STAGE2BS[st_name]
        acc_this = STAGE2ACC[st_name]
        logging.info(f"[{st_name}] bs={bs_this}, accum_steps={acc_this}, workers(T/V)={nw_train}/{nw_val}, fast_cudnn={'on' if args.fast_cudnn else 'off'}")

        train_ds = StageAwareMedicalDataset3D(
            image_paths=st_train_imgs, label_paths=st_train_lbls, active_sizes=st_ps,
            patches_per_case=4, cache_images=False, cache_labels=False, seed=123,
            augment=False, show_progress=True, npy_memmap=True, assume_normalized_images=True
        )
        val_ds = StageAwareMedicalDataset3D(
            image_paths=st_val_imgs, label_paths=st_val_lbls, active_sizes=st_ps,
            patches_per_case=2, cache_images=False, cache_labels=False, seed=123,
            augment=False, show_progress=True, npy_memmap=True, assume_normalized_images=True
        )

        g_train = torch.Generator(); g_train.manual_seed(123)
        g_val   = torch.Generator(); g_val.manual_seed(999)
        trainer.accum_steps = acc_this

        if len(st_ps) == 1:
            train_loader = make_stage_dataloader_single(
                train_ds, batch_size=bs_this, shuffle=True,
                num_workers=nw_train, pin_memory=(nw_train > 0),
                persistent_workers=(nw_train > 0), prefetch_factor=(2 if nw_train > 0 else 2),
                worker_init_fn=seed_worker, generator=g_train
            )
            val_loader = make_stage_dataloader_single(
                val_ds, batch_size=1, shuffle=False,
                num_workers=nw_val, pin_memory=(nw_val > 0),
                persistent_workers=(nw_val > 0), prefetch_factor=(1 if nw_val > 0 else 1),
                worker_init_fn=seed_worker, generator=g_val
            )
        else:
            train_multi = make_stage_dataloaders_multi(
                train_ds, batch_size_per_size=bs_this, shuffle=True,
                num_workers=nw_train, pin_memory=(nw_train > 0),
                persistent_workers=(nw_train > 0), prefetch_factor=(2 if nw_train > 0 else 2),
                worker_init_fn=seed_worker, generator=g_train
            )
            val_multi = make_stage_dataloaders_multi(
                val_ds, batch_size_per_size=1, shuffle=False,
                num_workers=nw_val, pin_memory=(nw_val > 0),
                persistent_workers=(nw_val > 0), prefetch_factor=(1 if nw_val > 0 else 1),
                worker_init_fn=seed_worker, generator=g_val
            )

        logging.info(f"== {st_name} | ps={st_ps} | epochs={num_epochs} | bs={bs_this} | acc={acc_this}")

        for k in range(start_k, num_epochs + 1):
            g_train.manual_seed(123 + start_epoch_global + k)
            ep = start_epoch_global + k

            if len(st_ps) == 1:
                train_loss = trainer.train_one_epoch(train_loader)
                val_loss, val_dice = trainer.validate(val_loader)
            else:
                train_losses_this = []
                val_losses_this, val_dices_this = [], []
                for ps, loader in train_multi.items():
                    trainer.accum_steps = acc_this
                    t_loss = trainer.train_one_epoch(loader)
                    train_losses_this.append(t_loss)
                for ps, loader in val_multi.items():
                    v_loss, v_dice = trainer.validate(loader)
                    if v_loss is not None and np.isfinite(v_loss):
                        val_losses_this.append(v_loss)
                    if np.isfinite(v_dice):
                        val_dices_this.append(v_dice)
                train_loss = float(np.mean(train_losses_this)) if train_losses_this else 0.0
                val_loss   = float(np.mean(val_losses_this))  if val_losses_this  else None
                val_dice   = float(np.mean(val_dices_this))  if val_dices_this  else 0.0

            # (9) sliding-window full-volume validation every 10 epochs (e.g., 9,19,29,...)
            if (ep % 10) == 9:
                # choose patch size for SW: take the largest available size in this stage for besteffective stride
                ps_for_sw = max(st_ps, key=lambda t: (t[0]*t[1]*t[2]))
                try:
                    sw_dice_plain = sliding_window_validate_cases(
                        model, st_val_imgs, st_val_lbls, patch_size=ps_for_sw,
                        overlap=0.5, device=device, use_channels_last=bool(args.channels_last), use_ema=None
                    )
                    sw_dice_ema = sliding_window_validate_cases(
                        model, st_val_imgs, st_val_lbls, patch_size=ps_for_sw,
                        overlap=0.5, device=device, use_channels_last=bool(args.channels_last), use_ema=ema
                    )
                    logging.info(f"[SW-VAL e{ep}] patch={ps_for_sw}  dice(model)={sw_dice_plain:.6f}  dice(EMA)={sw_dice_ema:.6f}")
                except Exception as e:
                    logging.warning(f"[SW-VAL e{ep}] failed: {e}")

            # Plateau scheduler (metric-based)
            if args.sched == "plateau":
                metric_value = (val_loss if args.plateau_metric == "val_loss" else -val_dice)
                if metric_value is not None and np.isfinite(metric_value):
                    scheduler.step(metric_value)

            train_losses.append(train_loss)
            val_losses.append(val_loss if val_loss is not None else float("nan"))
            val_dices.append(val_dice)

            is_best = val_dice > best_val
            if is_best:
                best_val = val_dice

            state = {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": trainer.scaler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val": best_val,
            }
            save_checkpoint(state, ckpt_dir, is_best=is_best)
            save_loss_history(train_losses, val_losses, val_dices, ckpt_dir)
            save_loss_graph(train_losses, val_losses, val_dices, ckpt_dir)

            try:
                lr_now = optimizer.param_groups[0]["lr"]
            except Exception:
                lr_now = float("nan")
            vloss_print = float("nan") if val_loss is None else val_loss
            logging.info(
                f"[{st_name}] ep {ep} | train {train_loss:.6f} | val {vloss_print:.6f} | "
                f"dice {val_dice:.6f} | lr {lr_now:.6e} | best {best_val:.6f}"
            )

        logging.info(f"<== END {st_name}")
        start_epoch_global += num_epochs


if __name__ == "__main__":
    main()
