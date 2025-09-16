import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from tqdm import tqdm

def normalize_ct(image):
    image = np.nan_to_num(image, nan=-1024.0, posinf=3071.0, neginf=-1024.0).astype(np.float32)
    image = np.clip(image, -1024.0, 3071.0)
    image = (image + 1024.0) / (3071.0 + 1024.0)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0)
    return image

def _roi_start_containing_point(pt, size, shape):
    (pz, py, px) = pt
    (sd, sh, sw) = size
    (D, H, W) = shape
    hd, hh, hw = sd // 2, sh // 2, sw // 2
    z0 = min(max(pz - hd, 0), max(D - sd, 0))
    y0 = min(max(py - hh, 0), max(H - sh, 0))
    x0 = min(max(px - hw, 0), max(W - sw, 0))
    return int(z0), int(y0), int(x0)

def _crop3d_by_start(arr, start, size):
    z0, y0, x0 = start
    sd, sh, sw = size
    return arr[z0:z0+sd, y0:y0+sh, x0:x0+sw]

def _load_3d_array(path, npy_memmap=True):
    p = path.lower()
    if p.endswith(".npy"):
        return np.load(path, mmap_mode="r" if npy_memmap else None)
    if p.endswith(".npz"):
        with np.load(path) as z:
            return z["arr"]
    if p.endswith(".nii") or p.endswith(".nii.gz"):
        if sitk is None:
            raise RuntimeError(f"SimpleITK not available to read NIfTI file: {path}")
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
    raise ValueError(f"Unsupported file extension: {path}")

class StageAwareMedicalDataset3D(Dataset):
    def __init__(self,
                 image_paths,
                 label_paths,
                 active_sizes=None,
                 patches_per_case=4,
                 cache_images=False,
                 cache_labels=False,
                 seed=123,
                 augment=False,
                 show_progress=True,
                 npy_memmap=True,
                 assume_normalized_images=True,
                 label_as_float=True):
        assert len(image_paths) == len(label_paths)
        self.image_paths = list(image_paths)
        self.label_paths = list(label_paths)
        self.n_cases = len(self.image_paths)
        self.allowed_sizes = [(64,256,256), (32,128,128), (16,64,64), (8,32,32)]
        self.active_sizes = list(active_sizes) if active_sizes is not None else [(64,256,256)]
        for ps in self.active_sizes:
            if ps not in self.allowed_sizes:
                raise ValueError(f"Active size {ps} not in allowed {self.allowed_sizes}")
        self.patches_per_case = int(patches_per_case)
        self.cache_images = bool(cache_images)
        self.cache_labels = bool(cache_labels)
        self.augment = bool(augment)
        self.show_progress = bool(show_progress)
        self.npy_memmap = bool(npy_memmap)
        self.assume_normalized_images = bool(assume_normalized_images)
        self.label_as_float = bool(label_as_float)
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.images = [None] * self.n_cases if self.cache_images else None
        self.labels = [None] * self.n_cases if self.cache_labels else None
        self.lbl_shapes = [None] * self.n_cases
        self.pos_idx = [None] * self.n_cases
        self._prime_label_index_cache()

    def set_active_sizes(self, sizes):
        if not isinstance(sizes, (list, tuple)) or len(sizes) < 1:
            raise ValueError("sizes must be non-empty list/tuple of (D,H,W).")
        for ps in sizes:
            if ps not in self.allowed_sizes:
                raise ValueError(f"size {ps} not in allowed {self.allowed_sizes}")
        self.active_sizes = list(sizes)

    def set_patches_per_case(self, k):
        self.patches_per_case = int(k)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

    def _prime_label_index_cache(self):
        iterator = range(self.n_cases)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Index labels (stage-aware)", ncols=100)
        for i in iterator:
            lbl = _load_3d_array(self.label_paths[i], npy_memmap=self.npy_memmap)
            lbl = (np.asarray(lbl) > 0).astype(np.uint8, copy=False)
            self.lbl_shapes[i] = lbl.shape
            pos = np.argwhere(lbl > 0)
            self.pos_idx[i] = pos.astype(np.int32)
            if self.cache_labels:
                self.labels[i] = np.array(lbl, copy=True)

    def __len__(self):
        return self.n_cases * max(1, self.patches_per_case)

    def _get_item_for_size(self, index, ps):
        case_idx = index % self.n_cases
        if self.cache_images and self.images[case_idx] is not None:
            img = self.images[case_idx]
        else:
            img = _load_3d_array(self.image_paths[case_idx], npy_memmap=self.npy_memmap)
            img = np.asarray(img)
            if self.cache_images:
                self.images[case_idx] = np.array(img, copy=True)
        if self.cache_labels and self.labels[case_idx] is not None:
            lbl = self.labels[case_idx]
        else:
            lbl = _load_3d_array(self.label_paths[case_idx], npy_memmap=self.npy_memmap)
            lbl = (np.asarray(lbl) > 0).astype(np.uint8, copy=False)
            if self.cache_labels:
                self.labels[case_idx] = np.array(lbl, copy=True)
        pos = self.pos_idx[case_idx]
        K = 10
        if pos is not None and pos.shape[0] > 0:
            tried = 0
            while True:
                j = self.rng.integers(0, pos.shape[0])
                pz, py, px = map(int, pos[j])
                start = _roi_start_containing_point((pz, py, px), ps, img.shape)
                img_p = _crop3d_by_start(img, start, ps)
                lbl_p = _crop3d_by_start(lbl, start, ps)
                if lbl_p.max() > 0:
                    break
                tried += 1
                if tried >= K:
                    break
        else:
            D, H, W = img.shape
            sd, sh, sw = ps
            z0 = self.rng.integers(0, max(1, D - sd + 1))
            y0 = self.rng.integers(0, max(1, H - sh + 1))
            x0 = self.rng.integers(0, max(1, W - sw + 1))
            img_p = _crop3d_by_start(img, (int(z0), int(y0), int(x0)), ps)
            lbl_p = _crop3d_by_start(lbl, (int(z0), int(y0), int(x0)), ps)
        if not self.assume_normalized_images:
            img_p = normalize_ct(img_p)
        if self.augment:
            img_p, lbl_p = self.augment_sample(img_p, lbl_p)
        img_t = torch.from_numpy(img_p.astype(np.float32, copy=False)).unsqueeze(0)
        if self.label_as_float:
            lbl_t = torch.from_numpy(lbl_p.astype(np.float32, copy=False))
        else:
            lbl_t = torch.from_numpy(lbl_p.astype(np.uint8, copy=False))
        return img_t, lbl_t

    def __getitem__(self, index):
        ps = self.active_sizes[self.rng.integers(0, len(self.active_sizes))]
        return self._get_item_for_size(index, ps)

    def augment_sample(self, image: np.ndarray, mask: np.ndarray):

        img0, msk0 = image, mask

        if self.py_rng.random() < 0.25:  # flip X
            image = np.flip(image, axis=2).copy()
            mask  = np.flip(mask,  axis=2).copy()
        if self.py_rng.random() < 0.25:  # flip Y
            image = np.flip(image, axis=1).copy()
            mask  = np.flip(mask,  axis=1).copy()
        if self.py_rng.random() < 0.10:  # flip Z 
            image = np.flip(image, axis=0).copy()
            mask  = np.flip(mask,  axis=0).copy()

        if self.py_rng.random() < 0.50:
            scale = float(self.rng.uniform(0.9, 1.1))
            shift = float(self.rng.uniform(-0.05, 0.05))
            image = image.astype(np.float32, copy=False) * scale + shift
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            image = np.clip(image, 0.0, 1.0)

        if self.py_rng.random() < 0.30:
            gamma = float(self.rng.uniform(0.7, 1.5))
            image = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
            image = np.power(image, gamma, dtype=np.float32)
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            image = np.clip(image, 0.0, 1.0)

        if mask.max() == 0:
            image, mask = img0, msk0

        return image, mask

class StageAwareDatasetView(Dataset):
    def __init__(self, base, fixed_size):
        self.base = base
        self.fixed_size = tuple(fixed_size)
        if self.fixed_size not in self.base.allowed_sizes:
            raise ValueError(f"fixed_size {self.fixed_size} not in {self.base.allowed_sizes}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return self.base._get_item_for_size(index, self.fixed_size)

def make_stage_dataloader_single(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=False,
    worker_init_fn=None,
    generator=None,
):
    use_mp = (num_workers or 0) > 0
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=(num_workers if use_mp else 0),
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    if use_mp:
        kwargs.update(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    return DataLoader(**kwargs)

def make_stage_dataloaders_multi(
    dataset,
    batch_size_per_size=2,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=False,
    worker_init_fn=None,
    generator=None,
):
    use_mp = (num_workers or 0) > 0
    loaders = {}
    for ps in dataset.allowed_sizes:
        if ps in dataset.active_sizes:
            view = StageAwareDatasetView(dataset, fixed_size=ps)
            kwargs = dict(
                dataset=view,
                batch_size=batch_size_per_size,
                shuffle=shuffle,
                num_workers=(num_workers if use_mp else 0),
                pin_memory=pin_memory,
                drop_last=drop_last,
                worker_init_fn=worker_init_fn,
                generator=generator,
            )
            if use_mp:
                kwargs.update(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
            loaders[ps] = DataLoader(**kwargs)
    return loaders
