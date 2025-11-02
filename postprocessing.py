import os
import numpy as np
import SimpleITK as sitk
from natsort import natsorted

def _is_nii(p):
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
    suf = ("_lbl","_pred","_post","_img")
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

def _read_pred_array(pred_path):
    if _is_nii(pred_path):
        pref = sitk.ReadImage(pred_path)
        return pref, sitk.GetArrayFromImage(pref)
    if pred_path.lower().endswith(".npy"):
        return None, np.load(pred_path)
    with np.load(pred_path) as z:
        return None, z["arr"]

def _find_img(pred_path, input_dir):
    base = os.path.basename(pred_path)
    core = _stem_core(base)
    cands = [
        f"{core}_img.nii.gz",
        f"{core}.nii.gz",
        f"{core}_img.nii",
        f"{core}.nii",
    ]
    for c in cands:
        p = os.path.join(input_dir, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no matching CT for {pred_path}")

def _connected_components(arr, ref):
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.CopyInformation(ref)
    return sitk.GetArrayFromImage(sitk.ConnectedComponent(img))

def _danielsson(bin_arr, ref):
    img = sitk.GetImageFromArray(bin_arr.astype(np.uint8))
    img.CopyInformation(ref)
    return sitk.GetArrayFromImage(sitk.DanielssonDistanceMap(img, False, False))

def _make_lung(img_path, hu_threshold):
    img = sitk.ReadImage(img_path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    lung = (arr <= float(hu_threshold)).astype(np.uint8)
    bin_img = sitk.GetImageFromArray(lung); bin_img.CopyInformation(img)
    cc = sitk.ConnectedComponent(bin_img)
    r = sitk.RelabelComponent(cc, sortByObjectSize=True)
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(r)
    size_x, size_y, size_z = r.GetSize()
    keeps = []
    for lab in stats.GetLabels():
        x0,y0,z0, sx,sy,sz = stats.GetBoundingBox(lab)
        x1 = x0+sx-1; y1 = y0+sy-1
        if not (x0==0 or x1==size_x-1 or y0==0 or y1==size_y-1):
            keeps.append((lab, stats.GetNumberOfPixels(lab)))
    keeps.sort(key=lambda t:t[1], reverse=True)
    keeps = keeps[:2]
    r_arr = sitk.GetArrayFromImage(r)
    out = np.zeros_like(r_arr, dtype=np.uint8)
    for i,(lab,_) in enumerate(keeps, start=1):
        out[r_arr==lab] = i
    if np.sum(out==2)>500000:
        out[out==2]=1
    else:
        out[out==2]=0
    return img, out

def _binary_dilate(arr, ref, r):
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.CopyInformation(ref)
    rr = int(max(r,0))
    dil = sitk.BinaryDilate(img, [rr, rr, rr])
    return sitk.GetArrayFromImage(dil)

def _select_surface(cc_arr, keep_mask, lung_bin, ref, topk, dilate_r):
    labs = np.unique(cc_arr[keep_mask]); labs = labs[labs>0]
    if labs.size==0: return keep_mask
    lung_d = _binary_dilate(lung_bin, ref, dilate_r)
    scores=[]
    for lab in labs:
        comp = (cc_arr==lab)
        score = int(np.sum(comp & (lung_d>0)))
        scores.append((score, lab))
    scores.sort(key=lambda t:t[0], reverse=True)
    chosen = {lab for score,lab in scores[:max(1,topk)] if score>0}
    if not chosen: return keep_mask
    m = np.zeros_like(keep_mask, bool)
    for lab in chosen:
        m |= (cc_arr==lab)
    return m

def _z_bounds(lung_bin):
    idx = np.where(lung_bin>0)[0]
    return (int(idx.min()), int(idx.max())) if idx.size>0 else None

def _apply_z_strict(cc_arr, keep_mask, lung_bin, ref):
    zb = _z_bounds(lung_bin)
    if zb is None:
        return np.zeros_like(keep_mask,bool)
    z0,z1 = zb
    z_spacing = ref.GetSpacing()[2]
    margin = int(np.ceil(20.0 / max(z_spacing, 1e-6)))
    if z1 - z0 + 1 <= 2*margin:
        return np.zeros_like(keep_mask,bool)
    z0m = z0 + margin
    z1m = z1 - margin
    labs = np.unique(cc_arr[keep_mask]); labs = labs[labs>0]
    m=np.zeros_like(keep_mask,bool)
    for lab in labs:
        comp=(cc_arr==lab)
        z_idx=np.where(comp.any(axis=(1,2)))[0]
        if z_idx.size>0 and (z_idx.min()>=z0m and z_idx.max()<=z1m):
            m |= comp
    return m

def _apply_z_clip(keep_mask, lung_bin):
    zb = _z_bounds(lung_bin)
    if zb is None:
        return np.zeros_like(keep_mask,bool)
    z0,z1 = zb
    m = np.zeros_like(keep_mask,bool); m[z0:z1+1] = True
    return keep_mask & m

def _make_out(pred_path, post_dir):
    b = os.path.basename(pred_path)
    if b.lower().endswith(".nii.gz"):
        return os.path.join(post_dir, b[:-7] + "_post.nii.gz")
    if b.lower().endswith(".nii"):
        return os.path.join(post_dir, b[:-4] + "_post.nii.gz")
    return os.path.join(post_dir, os.path.splitext(b)[0] + "_post.npy")

def _normalize_case_name(s):
    s = str(s)
    if s.endswith(".nii.gz"):
        s = s[:-7]
    elif s.endswith(".nii"):
        s = s[:-4]
    for suf in ("_pred","_prediction","_mask","_lbl","_img","_post"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    while s.endswith("_"):
        s = s[:-1]
    return s

def _read_points_from_excel(excel_path, sheet_name, case_name):
    import pandas as pd
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    cols = {c.lower(): c for c in df.columns.astype(str)}
    name_col = None
    for k in ("case_id","case"):
        if k in cols:
            name_col = cols[k]
            break
    if name_col is None:
        return []

    need = [k for k in ("x","y","z") if k in cols]
    if len(need) < 3:
        return []

    df["_norm_case"] = df[name_col].astype(str).apply(_normalize_case_name)
    rows = df[df["_norm_case"] == _normalize_case_name(case_name)]
    points = []
    if len(rows) == 0:
        return points

    for _, r in rows.iterrows():
        try:
            x = int(round(float(r[cols["x"]])))
            y = int(round(float(r[cols["y"]])))
            z = int(round(float(r[cols["z"]])))
            points.append((z, y, x))
        except Exception:
            continue
    return points

def _clip_center(z, y, x, shape):
    Z,Y,X = shape
    z = max(0, min(Z-1, int(z)))
    y = max(0, min(Y-1, int(y)))
    x = max(0, min(X-1, int(x)))
    return z,y,x

def _sphere_mask(shape, center, radius_vox):
    Z,Y,X = shape
    cz,cy,cx = center
    r = int(max(0, int(radius_vox)))
    z0 = max(0, cz - r); z1 = min(Z, cz + r + 1)
    y0 = max(0, cy - r); y1 = min(Y, cy + r + 1)
    x0 = max(0, cx - r); x1 = min(X, cx + r + 1)
    mask = np.zeros(shape, dtype=bool)
    zz = np.arange(z0, z1)[:, None, None]
    yy = np.arange(y0, y1)[None, :, None]
    xx = np.arange(x0, x1)[None, None, :]
    sub = (zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2 <= r**2
    mask[z0:z1, y0:y1, x0:x1] = sub
    return mask

def _select_one_cc_for_point(cc_arr, sphere_mask, excluded_labels):
    labs = np.unique(cc_arr[sphere_mask])
    labs = labs[(labs > 0) & (~np.isin(labs, list(excluded_labels)))]
    if labs.size == 0:
        return None
    # 겹침 픽셀 수 최대 라벨 선택
    best_lab = None
    best_overlap = -1
    for lab in labs:
        overlap = int(np.sum((cc_arr == lab) & sphere_mask))
        if overlap > best_overlap:
            best_overlap = overlap
            best_lab = int(lab)
    return best_lab

def _run_existing_pipeline(pred_arr, ref, lung_arr, dist_thresh, clip_z, z_strict, select_surface, topk, dilate_radius):
    pred_bin = (pred_arr>0).astype(np.uint8)
    cc_arr = _connected_components(pred_bin, ref)
    lung_bin = (lung_arr>0).astype(np.uint8)
    dist_arr = _danielsson(lung_bin, ref)
    keep = np.zeros_like(pred_bin, bool)
    for lab in range(1, int(cc_arr.max())+1):
        comp = (cc_arr==lab)
        if comp.any() and dist_arr[comp].min()<=dist_thresh:
            keep |= comp
    if z_strict:
        keep = _apply_z_strict(cc_arr, keep, lung_bin, ref)
    elif clip_z:
        keep = _apply_z_clip(keep, lung_bin)
    if select_surface:
        keep = _select_surface(cc_arr, keep, lung_bin, ref, topk, dilate_radius)
    return keep

def filter_inline(input_dir, out_dir, post_dir, hu_threshold, dist_thresh,
                  select_surface, topk, dilate_radius, clip_z, z_strict,
                  excel_path=None, sheet_name="Sheet1"):
    os.makedirs(post_dir, exist_ok=True)
    preds = natsorted([f for f in os.listdir(out_dir) if f.lower().endswith((".nii",".nii.gz",".npy",".npz"))])
    for case in preds:
        pred_path = os.path.join(out_dir, case)
        img_path = _find_img(pred_path, input_dir)
        img_ref, lung_arr = _make_lung(img_path, hu_threshold)
        pref, pred_arr = _read_pred_array(pred_path)
        ref = pref if pref is not None else img_ref

        pred_bin = (pred_arr > 0).astype(np.uint8)
        cc_arr = _connected_components(pred_bin, ref)

        excel_mode = excel_path is not None and len(str(excel_path).strip())>0 and os.path.exists(excel_path)
        keep = np.zeros_like(pred_bin, bool)

        if excel_mode:
            core = _stem_core(os.path.basename(pred_path))
            points = []
            try:
                points = _read_points_from_excel(excel_path, sheet_name, core)
            except Exception:
                points = []
            if len(points) > 0:
                z_spacing = float(ref.GetSpacing()[2]) if hasattr(ref, "GetSpacing") else 1.0
                radius_vox = int(np.ceil(20.0 / max(z_spacing, 1e-6)))

                chosen_labels = set()
                for (z, y, x) in points:
                    z, y, x = _clip_center(z, y, x, pred_arr.shape)
                    sphere = _sphere_mask(pred_arr.shape, (z, y, x), radius_vox)
                    lab = _select_one_cc_for_point(cc_arr, sphere, chosen_labels)
                    if lab is not None:
                        keep |= (cc_arr == lab)
                        chosen_labels.add(lab)

        if not np.any(keep):
            keep = _run_existing_pipeline(pred_arr, ref, lung_arr, dist_thresh, clip_z, z_strict, select_surface, topk, dilate_radius)

        if not np.any(keep):
            out = np.zeros_like(pred_arr, dtype=pred_arr.dtype)
        else:
            out = np.where(keep, pred_arr, 0).astype(pred_arr.dtype)

        ct = sitk.GetArrayFromImage(img_ref).astype(np.float32)
        out[ct <= hu_threshold] = 0

        out_path = _make_out(pred_path, post_dir)
        if _is_nii(out_path):
            o = sitk.GetImageFromArray(out); o.CopyInformation(ref); sitk.WriteImage(o, out_path)
        else:
            np.save(out_path, out)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--post_dir", type=str, required=True)
    ap.add_argument("--hu_threshold", type=float, default=-700)
    ap.add_argument("--dist_thresh", type=float, default=3)
    ap.add_argument("--select_surface", action="store_true")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--dilate_radius", type=int, default=1)
    ap.add_argument("--clip_z", action="store_true")
    ap.add_argument("--z_strict", action="store_true")
    ap.add_argument("--excel_path", type=str, default=None)
    ap.add_argument("--sheet_name", type=str, default="Sheet1")
    args=ap.parse_args()
    filter_inline(
        args.input_dir,
        args.out_dir if hasattr(args, "out_dir") else args.out_dir,
        args.post_dir if hasattr(args, "post_dir") else args.post_dir,
        args.hu_threshold,
        args.dist_thresh,
        args.select_surface,
        args.topk,
        args.dilate_radius,
        args.clip_z,
        args.z_strict,
        excel_path=args.excel_path,
        sheet_name=args.sheet_name
    )
