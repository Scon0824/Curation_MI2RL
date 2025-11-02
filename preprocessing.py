import os
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from natsort import natsorted
from sklearn.cluster import DBSCAN

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

def _read_image(nifti_path):
    img = sitk.ReadImage(nifti_path)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return img, arr, spacing, origin, direction

def _vox2world_zyx(vox_zyx, img):
    pts_mm = []
    for z,y,x in vox_zyx:
        p = img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        pts_mm.append((p[2], p[1], p[0]))
    return np.asarray(pts_mm, dtype=np.float32)

def _median_point_zyx(idx_zyx):
    z = int(np.round(np.median(idx_zyx[:,0])))
    y = int(np.round(np.median(idx_zyx[:,1])))
    x = int(np.round(np.median(idx_zyx[:,2])))
    return z,y,x

def _downsample_points(idx_zyx, max_points):
    if idx_zyx.shape[0] <= max_points:
        return idx_zyx
    sel = np.random.choice(idx_zyx.shape[0], size=max_points, replace=False)
    return idx_zyx[sel]

def cluster_points_dbscan(img, arr, eps_mm=3.0, min_samples=20, max_points=200000):
    fg = np.argwhere(arr > 0)
    if fg.size == 0:
        return []
    fg_ds = _downsample_points(fg, max_points)
    pts_mm_zyx = _vox2world_zyx(fg_ds, img)
    db = DBSCAN(eps=eps_mm, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(pts_mm_zyx[:, ::-1])
    uniq = np.unique(labels)
    uniq = [u for u in uniq if u != -1]
    clusters = []
    if len(uniq) == 0:
        return clusters
    centers_mm = []
    for lab in uniq:
        pts_lab = pts_mm_zyx[labels == lab]
        if pts_lab.shape[0] == 0:
            continue
        cz = np.median(pts_lab[:,0]); cy = np.median(pts_lab[:,1]); cx = np.median(pts_lab[:,2])
        centers_mm.append((lab, float(cz), float(cy), float(cx)))
    fg_full = _downsample_points(fg, max_points*2)
    fg_full_mm = _vox2world_zyx(fg_full, img)
    half_window_mm = eps_mm * 1.5
    for lab, cz, cy, cx in centers_mm:
        dz = fg_full_mm[:,0] - cz
        dy = fg_full_mm[:,1] - cy
        dx = fg_full_mm[:,2] - cx
        sel = (np.abs(dz) <= half_window_mm) & (np.abs(dy) <= half_window_mm) & (np.abs(dx) <= half_window_mm)
        cand = fg_full[sel]
        if cand.shape[0] == 0:
            cand_ds = fg_ds[labels == lab]
            if cand_ds.shape[0] == 0:
                continue
            z,y,x = _median_point_zyx(cand_ds)
        else:
            z,y,x = _median_point_zyx(cand)
        p_mm = img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        clusters.append({
            "cluster_id": int(lab),
            "z": int(z), "y": int(y), "x": int(x),
            "z_mm": float(p_mm[2]), "y_mm": float(p_mm[1]), "x_mm": float(p_mm[0]),
            "size_vox": int(np.sum(labels == lab))
        })
    clusters.sort(key=lambda d: d["size_vox"], reverse=True)
    for i,c in enumerate(clusters):
        c["cluster_id"] = i+1
    return clusters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True, type=str)
    ap.add_argument("--excel_path", required=True, type=str)
    ap.add_argument("--sheet_name", default="Sheet1", type=str)
    ap.add_argument("--eps_mm", type=float, default=3.0)
    ap.add_argument("--min_samples", type=int, default=20)
    ap.add_argument("--max_points", type=int, default=200000)
    args = ap.parse_args()
    cases = _list_nifti_recursive(args.gt_dir)
    rows = []
    for p in cases:
        try:
            img, arr, _, _, _ = _read_image(p)
            case_id = _stem_core(os.path.basename(p))
            clusters = cluster_points_dbscan(
                img, arr,
                eps_mm=args.eps_mm,
                min_samples=args.min_samples,
                max_points=args.max_points
            )
            if len(clusters) == 0:
                rows.append({
                    "case_id": case_id, "cluster_id": None,
                    "x": None, "y": None, "z": None,
                    "x_mm": None, "y_mm": None, "z_mm": None,
                    "size_vox": 0
                })
            else:
                for c in clusters:
                    rows.append({
                        "case_id": case_id,
                        "cluster_id": c["cluster_id"],
                        "x": c["x"], "y": c["y"], "z": c["z"],
                        "x_mm": c["x_mm"], "y_mm": c["y_mm"], "z_mm": c["z_mm"],
                        "size_vox": c["size_vox"]
                    })
        except Exception:
            rows.append({
                "case_id": _stem_core(os.path.basename(p)),
                "cluster_id": None, "x": None, "y": None, "z": None,
                "x_mm": None, "y_mm": None, "z_mm": None, "size_vox": 0
            })
    df = pd.DataFrame(rows, columns=["case_id","cluster_id","x","y","z","x_mm","y_mm","z_mm","size_vox"])
    os.makedirs(os.path.dirname(os.path.abspath(args.excel_path)), exist_ok=True)
    with pd.ExcelWriter(args.excel_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, sheet_name=args.sheet_name, index=False)

if __name__ == "__main__":
    main()
