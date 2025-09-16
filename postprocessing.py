import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def _is_nii(path):
    p = path.lower()
    return p.endswith(".nii") or p.endswith(".nii.gz")

def _list_images(img_dir):
    names = [f for f in os.listdir(img_dir) if f.lower().endswith((".nii",".nii.gz",".npy",".npz"))]
    names.sort()
    return [os.path.join(img_dir, n) for n in names]

def _read_img(path):
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

def _read_pred_for_image(img_path, pred_dir):
    base = os.path.basename(img_path)
    low = base.lower()
    if low.endswith(".nii.gz"):
        stem = base[:-7]
    elif low.endswith(".nii"):
        stem = base[:-4]
    else:
        stem = os.path.splitext(base)[0]
    cands = [
        os.path.join(pred_dir, stem + "_pred.nii.gz"),
        os.path.join(pred_dir, stem + "_pred.nii"),
        os.path.join(pred_dir, stem + "_pred.npy"),
        os.path.join(pred_dir, stem + ".nii.gz"),
        os.path.join(pred_dir, stem + ".npy"),
        os.path.join(pred_dir, base),
    ]
    pred_path = next((c for c in cands if os.path.exists(c)), None)
    if pred_path is None:
        raise FileNotFoundError(f"Prediction not found for {img_path}")
    if _is_nii(pred_path):
        ref = sitk.ReadImage(pred_path)
        arr = sitk.GetArrayFromImage(ref)
        return pred_path, ref, arr
    if pred_path.lower().endswith(".npy"):
        arr = np.load(pred_path, mmap_mode=None)
        return pred_path, None, np.asarray(arr)
    with np.load(pred_path) as z:
        arr = z["arr"]
    return pred_path, None, np.asarray(arr)

def _make_out_name(img_path, pred_path, out_dir):
    b = os.path.basename(pred_path)
    if "_img_pred" in b:
        name = b.replace("_img_pred", "_pred")
    else:
        low = b.lower()
        if low.endswith(".nii.gz"):
            name = b[:-7] + "_pred.nii.gz"
        elif low.endswith(".nii"):
            name = b[:-4] + "_pred.nii.gz"
        elif low.endswith(".npy") or low.endswith(".npz"):
            name = os.path.splitext(b)[0] + "_pred.npy"
        else:
            name = os.path.splitext(b)[0] + "_pred"
    if _is_nii(img_path) and not name.lower().endswith((".nii",".nii.gz")):
        stem = os.path.splitext(name)[0]
        if stem.endswith(".nii"): stem = stem[:-4]
        name = stem + ".nii.gz"
    return os.path.join(out_dir, name)

def _save_lbl(ref_img, arr, out_path):
    arr = arr.astype(np.uint8, copy=False)
    if _is_nii(out_path):
        img = sitk.GetImageFromArray(arr)
        if ref_img is not None:
            img.CopyInformation(ref_img)
        sitk.WriteImage(img, out_path)
    else:
        np.save(out_path, arr)

def _binarize(pred_arr, thr=0.5):
    a = np.asarray(pred_arr)
    if a.dtype != np.uint8:
        a = (a >= thr).astype(np.uint8)
    return a

def _mask_by_hu(img_arr, pred_arr, hu_thresh=-600.0):
    img_arr = np.asarray(img_arr)
    pred_arr = np.asarray(pred_arr)
    amin = float(np.nanmin(img_arr))
    amax = float(np.nanmax(img_arr))
    if amin >= -0.2 and amax <= 1.2:
        thr = (hu_thresh + 1024.0) / (3071.0 + 1024.0)
    else:
        thr = hu_thresh
    pred_arr = pred_arr.copy()
    pred_arr[img_arr < thr] = 0
    return pred_arr

def _remove_small_components(pred_bin, ref_img_for_meta, min_voxels=50):
    img = sitk.GetImageFromArray(pred_bin.astype(np.uint8, copy=False))
    if ref_img_for_meta is not None:
        img.CopyInformation(ref_img_for_meta)
    cc = sitk.ConnectedComponent(img)
    rel = sitk.RelabelComponent(cc, minimumObjectSize=int(min_voxels))
    out = sitk.BinaryThreshold(rel, lowerThreshold=1, upperThreshold=2**31-1, insideValue=1, outsideValue=0)
    return sitk.GetArrayFromImage(out).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--post_dir", type=str, required=True)
    ap.add_argument("--min_hu", type=float, default=-600.0)
    ap.add_argument("--min_cluster", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.post_dir, exist_ok=True)
    imgs = _list_images(args.input_dir)

    for ip in tqdm(imgs, desc="postprocess", ncols=100, leave=False):
        img_ref, img_arr = _read_img(ip)
        pred_path, pred_ref, pred_arr = _read_pred_for_image(ip, args.out_dir)
        pred_bin = _binarize(pred_arr, thr=0.5)
        pred_hu = _mask_by_hu(img_arr, pred_bin, hu_thresh=args.min_hu)
        pred_cc = _remove_small_components(
            pred_hu,
            img_ref if img_ref is not None else pred_ref,
            min_voxels=args.min_cluster,
        )
        out_path = _make_out_name(ip, pred_path, args.post_dir)
        _save_lbl(img_ref if img_ref is not None else pred_ref, pred_cc, out_path)

if __name__ == "__main__":
    main()
