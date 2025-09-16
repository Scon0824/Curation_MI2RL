import os
import csv
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

SUPP_EXT = (".nii", ".nii.gz", ".npy", ".npz")

def _is_nii(path):
    p = path.lower()
    return p.endswith(".nii") or p.endswith(".nii.gz")

def _list_files(d):
    names = [f for f in os.listdir(d) if f.lower().endswith(SUPP_EXT)]
    names.sort()
    return [os.path.join(d, n) for n in names]

def _stem_core(base):
    b = base
    lb = b.lower()
    if lb.endswith(".nii.gz"):
        b = b[:-7]
    elif lb.endswith(".nii"):
        b = b[:-4]
    else:
        b = os.path.splitext(b)[0]
    for suf in ("_lbl", "_pred", "_post"):
        if b.endswith(suf):
            b = b[: -len(suf)]
    return b

def _read_arr(path):
    if _is_nii(path):
        ref = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(ref)
        return ref, np.asarray(arr)
    if path.lower().endswith(".npy"):
        arr = np.load(path, mmap_mode=None)
        return None, np.asarray(arr)
    with np.load(path) as z:
        arr = z["arr"]
    return None, np.asarray(arr)

def _pair_gt_for_post(post_path, gt_dir):
    b = os.path.basename(post_path)
    core = _stem_core(b)
    core = core.replace("_pred", "")
    cands = [
        os.path.join(gt_dir, core + ".nii.gz"),
        os.path.join(gt_dir, core + ".nii"),
        os.path.join(gt_dir, core + ".npy"),
        os.path.join(gt_dir, core + ".npz"),
        os.path.join(gt_dir, core + "_lbl.nii.gz"),
        os.path.join(gt_dir, core + "_lbl.nii"),
        os.path.join(gt_dir, core + "_lbl.npy"),
        os.path.join(gt_dir, core + "_lbl.npz"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None

def _ensure_binary(a):
    a = np.asarray(a)
    return (a > 0).astype(np.uint8)  # <— FIX: 255 → 1, floats → 0/1, bool → 0/1

def _dice_bin(pred, gt):
    pred = _ensure_binary(pred)
    gt   = _ensure_binary(gt)
    inter = int(np.sum((pred == 1) & (gt == 1)))
    s = int(pred.sum()) + int(gt.sum())
    if s == 0:
        return 1.0
    return (2.0 * inter) / max(1, s)

def _voxel_recall(pred, gt):
    pred = _ensure_binary(pred)
    gt   = _ensure_binary(gt)
    tp = int(np.sum((pred == 1) & (gt == 1)))
    g  = int(gt.sum())
    if g == 0:
        return 1.0
    return tp / g

def _component_recall(pred, gt):
    pred = _ensure_binary(pred)
    gt   = _ensure_binary(gt)
    gt_img   = sitk.GetImageFromArray(gt.astype(np.uint8, copy=False))
    pred_img = sitk.GetImageFromArray(pred.astype(np.uint8, copy=False))
    cc = sitk.ConnectedComponent(gt_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    n_labels = len(stats.GetLabels())
    if n_labels == 0:
        return 1.0
    over = 0
    for lab in stats.GetLabels():
        z0, y0, x0, dz, dy, dx = stats.GetBoundingBox(lab)
        z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
        gt_crop   = gt[z0:z1, y0:y1, x0:x1]
        pred_crop = pred[z0:z1, y0:y1, x0:x1]
        if np.any((gt_crop == 1) & (pred_crop == 1)):
            over += 1
    return over / n_labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--post_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    posts = _list_files(args.post_dir)
    rows = []
    dvals, rvals, cvals = [], [], []

    for pp in tqdm(posts, desc="evaluate", ncols=100, leave=False):
        gp = _pair_gt_for_post(pp, args.gt_dir)
        if gp is None:
            print(f"[warn] GT not found for: {pp}")
            continue
        _, post = _read_arr(pp)
        _, gt   = _read_arr(gp)
        if post.shape != gt.shape:
            print(f"[warn] shape mismatch, skip: post={post.shape} gt={gt.shape} | {os.path.basename(pp)}")
            continue

        post_bin = _ensure_binary(post)
        gt_bin   = _ensure_binary(gt)

        d = _dice_bin(post_bin, gt_bin)
        r = _voxel_recall(post_bin, gt_bin)

        dvals.append(d); rvals.append(r)
        tp = int(np.sum((post_bin==1) & (gt_bin==1)))
        fp = int(np.sum((post_bin==1) & (gt_bin==0)))
        fn = int(np.sum((post_bin==0) & (gt_bin==1)))
        rows.append([
            os.path.basename(pp),
            int(gt_bin.sum()),
            int(post_bin.sum()),
            tp, fp, fn,
            f"{d:.4f}",
            f"{r:.4f}", 
        ])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case","gt_voxels","post_voxels","tp","fp","fn","dice","voxel_recall"])
        for r in rows:
            w.writerow(r)

    if dvals:
        md = float(np.mean(dvals)); sd = float(np.std(dvals))
        mr = float(np.mean(rvals)); sr = float(np.std(rvals))
        print(f"Dice mean±std: {md:.4f} ± {sd:.4f}  |  Recall(mean±std): {mr:.4f} ± {sr:.4f}")
    else:
        print("No valid pairs evaluated.")

if __name__ == "__main__":
    main()
