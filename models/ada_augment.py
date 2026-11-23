import torch
import torch.nn.functional as F


def ada_augment(images: torch.Tensor, probability: float) -> torch.Tensor:
    # Apply per-image geometric and color augmentations at an adaptive probability

    # Ensure correct arguments
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("ADA expects images shaped [batch, 3, height, width].")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("ADA probability must be in the range [0, 1].")
    if probability == 0.0:
        return images

    batch_size, _, height, width = images.shape
    device = images.device
    dtype = images.dtype

    flip = torch.rand(batch_size, device=device) < probability
    translate = torch.rand(batch_size, device=device) < probability
    max_shift_x = max(1, round(width * 0.05))
    max_shift_y = max(1, round(height * 0.05))
    shift_x = torch.randint(
        -max_shift_x, max_shift_x + 1, (batch_size,), device=device
    )
    shift_y = torch.randint(
        -max_shift_y, max_shift_y + 1, (batch_size,), device=device
    )
    shift_x = torch.where(translate, shift_x, torch.zeros_like(shift_x))
    shift_y = torch.where(translate, shift_y, torch.zeros_like(shift_y))

    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = 1.0 - 2.0 * flip.to(dtype)
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = 2.0 * shift_x.to(dtype) / width
    theta[:, 1, 2] = 2.0 * shift_y.to(dtype) / height
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    augmented = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )

    blur_mask = (torch.rand(batch_size, device=device) < probability).view(
        batch_size, 1, 1, 1
    )
    blur_kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 16.0
    blur_kernel = blur_kernel.expand(3, 1, 3, 3)
    blurred = F.conv2d(
        F.pad(augmented, (1, 1, 1, 1), mode="reflect"),
        blur_kernel,
        groups=3,
    )
    augmented = torch.where(blur_mask, blurred, augmented)

    brightness_mask = (torch.rand(batch_size, device=device) < probability).view(
        batch_size, 1, 1, 1
    )
    brightness = torch.rand(
        batch_size, 1, 1, 1, device=device, dtype=dtype
    ) * 0.4 - 0.2
    augmented = augmented + torch.where(
        brightness_mask, brightness, torch.zeros_like(brightness)
    )

    contrast_mask = (torch.rand(batch_size, device=device) < probability).view(
        batch_size, 1, 1, 1
    )
    contrast = 0.8 + 0.4 * torch.rand(
        batch_size, 1, 1, 1, device=device, dtype=dtype
    )
    contrast = torch.where(contrast_mask, contrast, torch.ones_like(contrast))
    image_mean = augmented.mean(dim=(1, 2, 3), keepdim=True)
    augmented = (augmented - image_mean) * contrast + image_mean

    saturation_mask = (torch.rand(batch_size, device=device) < probability).view(
        batch_size, 1, 1, 1
    )
    saturation = 0.8 + 0.4 * torch.rand(
        batch_size, 1, 1, 1, device=device, dtype=dtype
    )
    saturation = torch.where(saturation_mask, saturation, torch.ones_like(saturation))
    grayscale = (
        0.299 * augmented[:, 0:1]
        + 0.587 * augmented[:, 1:2]
        + 0.114 * augmented[:, 2:3]
    )
    augmented = grayscale + (augmented - grayscale) * saturation

    return augmented.clamp(-1.0, 1.0)
