import os
import re
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from natsort import natsorted
from tqdm import tqdm

def _is_nifti(p):
    pl = p.lower()
    return pl.endswith(".nii") or pl.endswith(".nii.gz")

def _strip_ext(b):
    lb = b.lower()
    if lb.endswith(".nii.gz"):
        return b[:-7]
    if lb.endswith(".nii"):
        return b[:-4]
    return os.path.splitext(b)[0]

def _stem_core(b):
    b = _strip_ext(b)
    suf = ("_lbl","_label","_mask","_gt","_img","_pred","_post")
    changed = True
    while changed:
        changed = False
        for s in suf:
            if b.endswith(s):
                b = b[: -len(s)]
                changed = True
    while b.endswith("_"):
        b = b[:-1]
    return b

def _list_nifti_recursive(root_dir):
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            p = os.path.join(dirpath, f)
            if _is_nifti(p):
                out.append(p)
    return natsorted(out)

def _read_image_from_nifti(nifti_path):
    img = sitk.ReadImage(nifti_path)
    arr = sitk.GetArrayFromImage(img)
    case_id = _stem_core(os.path.basename(nifti_path))
    return case_id, img, arr

def _median_xyz_from_mask(arr, include_zero=False):
    mask = (arr >= 0) if include_zero else (arr > 0)
    if not np.any(mask):
        return None
    z_idx, y_idx, x_idx = np.where(mask)
    z_med = int(np.round(np.median(z_idx)))
    y_med = int(np.round(np.median(y_idx)))
    x_med = int(np.round(np.median(x_idx)))
    return (z_med, y_med, x_med)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", required=True, type=str)
    p.add_argument("--excel_path", required=True, type=str)
    p.add_argument("--sheet_name", default="Sheet1", type=str)
    p.add_argument("--include_zero", action="store_true")
    args = p.parse_args()

    cases = _list_nifti_recursive(args.gt_dir)
    rows = []

    for nifti_path in tqdm(cases, desc="Computing medians", unit="case"):
        try:
            case_id, img, arr = _read_image_from_nifti(nifti_path)
            med = _median_xyz_from_mask(arr, include_zero=args.include_zero)
            if med is None:
                z = y = x = None
            else:
                z, y, x = med
            rows.append({"case_id": case_id, "x": x, "y": y, "z": z})
        except Exception as e:
            tqdm.write(f"[warn] {os.path.basename(nifti_path)}: {e}")
            rows.append({"case_id": _stem_core(os.path.basename(nifti_path)), "x": None, "y": None, "z": None})

    df = pd.DataFrame(rows, columns=["case_id","x","y","z"])
    os.makedirs(os.path.dirname(os.path.abspath(args.excel_path)), exist_ok=True)
    with pd.ExcelWriter(args.excel_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, sheet_name=args.sheet_name, index=False)

if __name__ == "__main__":
    main()
