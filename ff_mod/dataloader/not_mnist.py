from ff_mod.dataloader.base import DataLoaderExtractor

import torch
import deeplake

try:
    from datasets import get_dataset_split_names, load_dataset
except Exception:
    get_dataset_split_names = None
    load_dataset = None


def _open_deeplake_dataset(uri: str):
    # Prefer read-only opening for hub datasets.
    if hasattr(deeplake, "open_read_only"):
        return deeplake.open_read_only(uri)

    # Deep Lake 4.x replaced load() by open(), while 3.x still uses load().
    if hasattr(deeplake, "open"):
        return deeplake.open(uri)

    load_fn = getattr(deeplake, "load", None)
    if load_fn is None:
        raise RuntimeError("DeepLake API does not expose open/open_read_only/load")
    return load_fn(uri)


def _resolve_hf_split(split, available_splits):
    if split in available_splits:
        return split

    if split == "train" and "train" in available_splits:
        return "train"
    if split == "test" and "test" in available_splits:
        return "test"
    if split == "val" and "validation" in available_splits:
        return "validation"

    if "test" in available_splits:
        return "test"
    if "train" in available_splits:
        return "train"
    return available_splits[0]


class _HFNotMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]

        image = sample["image"]
        if hasattr(image, "convert"):
            image = image.convert("L")

        x = self.transform(image)
        y = int(sample.get("label", 0))

        return x, y

class NotMNIST(DataLoaderExtractor):
    
    def __init__(self, batch_size = 64, large = False, **kwargs):
        super().__init__(batch_size=batch_size)
        self.version = "small" if not large else "large"
    
    def load_dataloader(self, download = False, split = None, **kwargs):
        split = "test" if split is None else split

        hf_error = None
        if load_dataset is not None and get_dataset_split_names is not None:
            try:
                splits = get_dataset_split_names("anubhavmaity/notMNIST")
                hf_split = _resolve_hf_split(split, splits)
                hf_ds = load_dataset("anubhavmaity/notMNIST", split=hf_split)

                dataset = _HFNotMNISTDataset(hf_ds, self.transform)
                self.dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,
                )
                return
            except Exception as exc:
                hf_error = exc

        try:
            ds = _open_deeplake_dataset('hub://activeloop/not-mnist-' + self.version)
            pytorch_loader = getattr(ds, "pytorch")
            self.dataloader = pytorch_loader(
                num_workers=0,
                batch_size=self.batch_size,
                shuffle=True,
                transform={'images': self.transform, 'labels': None}
            )
            return
        except Exception as deeplake_error:
            if hf_error is not None:
                raise RuntimeError(
                    f"Failed to load notMNIST from HuggingFace and DeepLake. "
                    f"HF error: {hf_error}. DeepLake error: {deeplake_error}"
                ) from deeplake_error
            raise
    
        