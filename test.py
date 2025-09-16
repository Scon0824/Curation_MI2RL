import os
import argparse
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from model import MultiScaleRoutedResUNet3D_DS

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def normalize_ct(image):
    image = np.nan_to_num(image, nan=-1024.0, posinf=3071.0, neginf=-1024.0).astype(np.float32)
    image = np.clip(image, -1024.0, 3071.0)
    image = (image + 1024.0) / (3071.0 + 1024.0)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0)
    return image

def _list_inputs(input_dir):
    names = [f for f in os.listdir(input_dir) if f.lower().endswith((".nii", ".nii.gz", ".npy", ".npz"))]
    names.sort()
    return [os.path.join(input_dir, n) for n in names]

def _is_nii(path):
    p = path.lower()
    return p.endswith(".nii") or p.endswith(".nii.gz")

def _read_arr_with_header(path):
    if _is_nii(path):
        ref = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(ref)
        return ref, arr
    if path.lower().endswith(".npy"):
        arr = np.load(path, mmap_mode=None)
        return None, np.asarray(arr)
    with np.load(path) as z:
        arr = z["arr"]
    return None, np.asarray(arr)

def _make_pred_name(path, out_dir):
    base = os.path.basename(path)
    low = base.lower()
    if low.endswith(".nii.gz"):
        name = base[:-7] + "_pred.nii.gz"
    elif low.endswith(".nii"):
        name = base[:-4] + "_pred.nii.gz"
    elif low.endswith(".npy"):
        name = os.path.splitext(base)[0] + "_pred.npy"
    elif low.endswith(".npz"):
        name = os.path.splitext(base)[0] + "_pred.npy"
    else:
        name = os.path.splitext(base)[0] + "_pred"
    return os.path.join(out_dir, name)

def _save_like(ref_img, arr, out_path):
    if ref_img is None:
        np.save(out_path, arr.astype(np.uint8, copy=False))
        return
    out = sitk.GetImageFromArray(arr.astype(np.uint8, copy=False))
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, out_path)

def _allowed_map():
    return {
        "0": (64, 256, 256),
        "1": (32, 128, 128),
        "2": (16, 64, 64),
        "3": (8, 32, 32),
    }

def _allowed_list():
    m = _allowed_map()
    return [m["0"], m["1"], m["2"], m["3"]]

def _parse_patch_mode(patch_mode):
    pm = str(patch_mode).strip().lower()
    if pm == "whole":
        return "whole", _allowed_list()
    m = _allowed_map()
    if pm in m:
        return "single", [m[pm]]
    raise ValueError("patch must be one of: 0,1,2,3,whole")

def _pad_min(vol, ps):
    dz = max(0, ps[0] - vol.shape[0])
    dy = max(0, ps[1] - vol.shape[1])
    dx = max(0, ps[2] - vol.shape[2])
    if dz == 0 and dy == 0 and dx == 0:
        return vol, (0, 0, 0, 0, 0, 0)
    z0 = dz // 2; z1 = dz - z0
    y0 = dy // 2; y1 = dy - y0
    x0 = dx // 2; x1 = dx - x0
    vol_p = np.pad(vol, ((z0, z1), (y0, y1), (x0, x1)), mode="edge")
    return vol_p, (z0, z1, y0, y1, x0, x1)

def _unpad(vol, pads):
    z0, z1, y0, y1, x0, x1 = pads
    if z0 + z1 > 0: vol = vol[z0:vol.shape[0] - z1]
    if y0 + y1 > 0: vol = vol[:, y0:vol.shape[1] - y1]
    if x0 + x1 > 0: vol = vol[:, :, x0:vol.shape[2] - x1]
    return vol

def _gaussian_weight_3d(ps):
    z, y, x = ps
    zz = np.linspace(-1, 1, z, dtype=np.float32)
    yy = np.linspace(-1, 1, y, dtype=np.float32)
    xx = np.linspace(-1, 1, x, dtype=np.float32)
    Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
    D2 = Z * Z + Y * Y + X * X
    w = np.exp(-4.5 * D2)
    w = w / np.max(w)
    return torch.from_numpy(w.astype(np.float32))

def replace_instancenorm_with_groupnorm(model, num_groups=16):
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.InstanceNorm3d):
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            gn = torch.nn.GroupNorm(num_groups=num_groups, num_channels=module.num_features, affine=True)
            setattr(parent, attr_name, gn)
    return model

def _load_model(ckpt_path, device, use_groupnorm=True, gn_groups=16):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = {"in_channels": 1, "out_channels": 1, "base_features": 32, "upsample_to_input": True}
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_args", None), dict):
        for k, v in ckpt["model_args"].items():
            args[k] = v
    m = MultiScaleRoutedResUNet3D_DS(**args)
    if use_groupnorm:
        m = replace_instancenorm_with_groupnorm(m, num_groups=gn_groups)
    if isinstance(ckpt, dict):
        state = ckpt.get("model", None)
        if state is None: state = ckpt.get("model_state", None)
        if state is None: state = ckpt.get("state_dict", None)
        if state is None: state = ckpt.get("model_state_dict", None)
        if state is not None:
            m.load_state_dict(state, strict=True)
        else:
            try:
                m.load_state_dict(ckpt, strict=True)
            except Exception:
                pass
    m.to(device).eval()
    return m

def _slide_once(model, vol_np, patch, overlap, device):
    D, H, W = vol_np.shape
    pD, pH, pW = patch
    sD = max(1, int(pD * (1 - overlap)))
    sH = max(1, int(pH * (1 - overlap)))
    sW = max(1, int(pW * (1 - overlap)))
    w = _gaussian_weight_3d(patch).to(device)
    acc = torch.zeros((D, H, W), dtype=torch.float32, device=device)
    wsum = torch.zeros((D, H, W), dtype=torch.float32, device=device)
    z_st = list(range(0, max(1, D - pD + 1), sD))
    y_st = list(range(0, max(1, H - pH + 1), sH))
    x_st = list(range(0, max(1, W - pW + 1), sW))
    if z_st[-1] != D - pD: z_st.append(max(0, D - pD))
    if y_st[-1] != H - pH: y_st.append(max(0, H - pH))
    if x_st[-1] != W - pW: x_st.append(max(0, W - pW))
    with torch.no_grad():
        for z0 in z_st:
            for y0 in y_st:
                for x0 in x_st:
                    patch_np = vol_np[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW]
                    ten = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
                    outs = model(ten, upsample_to_input=True)
                    logits = outs[0][0, 0] if isinstance(outs, list) else outs[0, 0]
                    acc[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW] += logits * w
                    wsum[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW] += w
    acc = acc / torch.clamp(wsum, min=1e-8)
    return acc

def _infer_with_mode(model, arr, patch_mode, overlap, device):
    mode, patches = _parse_patch_mode(patch_mode)
    arrf = arr.astype(np.float32, copy=False)
    amin = float(np.nanmin(arrf))
    amax = float(np.nanmax(arrf))
    if amin >= -1e-6 and amax <= 1.0 + 1e-6:
        arr_n = np.nan_to_num(arrf, nan=0.0, posinf=1.0, neginf=0.0)
    else:
        arr_n = normalize_ct(arrf)
    logits_sum = None
    cnt = 0
    for ps in patches:
        vol_p, pads = _pad_min(arr_n, ps)
        logits_p = _slide_once(model, vol_p, ps, overlap, device).detach().cpu().numpy()
        logits = _unpad(logits_p, pads)
        if logits_sum is None:
            logits_sum = logits
            cnt = 1
        else:
            logits_sum = logits_sum + logits
            cnt += 1
    logits_mean = logits_sum / max(1, cnt)
    prob = _sigmoid(logits_mean)
    pred = (prob >= 0.5).astype(np.uint8)
    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--patch", type=str, required=True)
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.ckpt, device, use_groupnorm=True, gn_groups=16)
    paths = _list_inputs(args.input_dir)

    for ip in tqdm(paths, desc="infer", ncols=100, leave=False):
        ref, arr = _read_arr_with_header(ip)
        pred = _infer_with_mode(model, arr, args.patch, args.overlap, device)
        out_pred = _make_pred_name(ip, args.out_dir)
        _save_like(ref, pred, out_pred)

if __name__ == "__main__":
    main()
