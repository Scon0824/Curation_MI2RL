import os
import argparse
import SimpleITK as sitk
from natsort import natsorted
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def convert_case(case_info):
    dcm_root, out_dir, case = case_info
    dcm_path = os.path.join(dcm_root, case)
    nii_path = os.path.join(out_dir, f"{case}.nii.gz")
    try:
        os.makedirs(out_dir, exist_ok=True)
        reader = sitk.ImageSeriesReader()

        series_ids = reader.GetGDCMSeriesIDs(dcm_path)
        if not series_ids:
            raise RuntimeError(f"No DICOM series found in: {dcm_path}")

        best_sid = None
        best_files = None
        best_len = -1
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(dcm_path, sid)
            if len(files) > best_len:
                best_len = len(files)
                best_sid = sid
                best_files = files

        reader.SetFileNames(best_files)
        img = reader.Execute()
        sitk.WriteImage(img, nii_path)
        return True, case, ""
    except Exception as e:
        return False, case, str(e)

def convert_dicom_to_nifti(dcm_root, out_dir, workers=16):
    os.makedirs(out_dir, exist_ok=True)
    all_entries = natsorted(os.listdir(dcm_root))
    cases = [name for name in all_entries if os.path.isdir(os.path.join(dcm_root, name))]

    case_infos = [(dcm_root, out_dir, case) for case in cases]
    if not case_infos:
        print(f"[warn] No case folders under: {dcm_root}")
        return

    if workers <= 0:
        workers = max(1, cpu_count() // 2)
    with Pool(processes=workers) as pool:
        results = list(tqdm(pool.imap(convert_case, case_infos), total=len(case_infos), desc="dcm->nii", ncols=100))

    ok = sum(1 for s, _, _ in results if s)
    fail = [(c, msg) for s, c, msg in results if not s]
    print(f"[done] converted: {ok} / {len(results)}")
    if fail:
        print("[errors]")
        for c, msg in fail[:20]:
            print(f"  {c}: {msg}")
        if len(fail) > 20:
            print(f"  ... and {len(fail)-20} more")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcm_dir", type=str, required=True)
    ap.add_argument("--nii_dir", type=str, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    convert_dicom_to_nifti(args.dcm_dir, args.nii_dir, workers=args.workers)

if __name__ == "__main__":
    main()
