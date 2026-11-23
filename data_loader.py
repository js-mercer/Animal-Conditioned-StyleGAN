from pathlib import Path

import kagglehub
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATASET_HANDLE = "dimensi0n/afhq-512"
CLASS_NAMES = {"cat", "dog", "wild"}


class AFHQImageFolder(datasets.ImageFolder):
    """ImageFolder that ignores non-class directories in the extracted dataset."""

    def find_classes(self, directory):
        classes = sorted(
            class_name
            for class_name in CLASS_NAMES
            if (Path(directory) / class_name).is_dir()
        )
        if set(classes) != CLASS_NAMES:
            raise FileNotFoundError(
                f"Expected class folders {sorted(CLASS_NAMES)} in {directory}, "
                f"but found {classes}."
            )
        return classes, {class_name: index for index, class_name in enumerate(classes)}


def _find_training_directory(dataset_path: Path) -> Path:
    """Find the directory whose immediate subdirectories are the AFHQ classes."""
    candidates = []
    directories = [dataset_path]
    directories.extend(path for path in dataset_path.rglob("*") if path.is_dir())

    for directory in directories:
        child_dirs = {path.name.lower() for path in directory.iterdir() if path.is_dir()}
        if CLASS_NAMES.issubset(child_dirs):
            candidates.append(directory)

    train_candidates = [path for path in candidates if path.name.lower() == "train"]
    if len(train_candidates) == 1:
        return train_candidates[0]
    if len(train_candidates) > 1:
        raise ValueError(
            f"Found multiple AFHQ training directories under {dataset_path}: "
            f"{train_candidates}"
        )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a directory containing cat, dog, and wild class folders "
            f"under the downloaded dataset at {dataset_path}."
        )
    raise ValueError(
        f"Found multiple directories containing AFHQ class folders under "
        f"{dataset_path}, but none is named 'train': {candidates}"
    )


def get_afhqv2_loaders(
    batch_size, data_dir="./data", num_workers=0, random_horizontal_flip=True
):
    """Download AFHQv2 with KaggleHub and return its shuffled training loader."""
    downloaded_path = Path(
        kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(data_dir))
    )
    training_dir = _find_training_directory(downloaded_path)

    # Transform data to tensor and normalise to [-1, 1] range.
    transform_steps = [transforms.Resize((128, 128))]
    if random_horizontal_flip:
        transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    transform = transforms.Compose(transform_steps)

    trainset = AFHQImageFolder(root=str(training_dir), transform=transform)
    if set(trainset.classes) != CLASS_NAMES:
        raise ValueError(
            f"Expected AFHQ classes {sorted(CLASS_NAMES)}, "
            f"but found {trainset.classes} in {training_dir}."
        )

    # Return a shuffled DataLoader
    return DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )