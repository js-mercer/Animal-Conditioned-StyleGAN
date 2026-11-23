import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from data_loader import get_afhqv2_loaders
from models.ada_augment import ada_augment
from models.discriminator.stylegan_discriminator import StyleGANDiscriminator
from models.generator.stylegan_generator import StyleGANGenerator


NUM_CLASSES = 3
IMAGE_SIZE = 128
LATENT_DIM = 512
SAMPLES_PER_CLASS = 16
SAMPLES_PER_ROW = 8


def parse_args() -> argparse.Namespace:
    """Parse training settings and output locations."""
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train a class-conditioned StyleGAN on AFHQ."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes; try 2 or 4 in Colab.",
    )
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--ema-beta", type=float, default=0.995)
    parser.add_argument("--mix-prob", type=float, default=0.9)
    parser.add_argument("--log-interval", type=int, default=64)
    parser.add_argument(
        "--r1-gamma",
        type=float,
        default=0.16,
        help="R1 gradient penalty strength (default: 0.16).",
    )
    parser.add_argument(
        "--r1-interval",
        type=int,
        default=16,
        help="Apply R1 once every N discriminator updates (default: 16).",
    )
    parser.add_argument(
        "--ada",
        action="store_true",
        help="Enable adaptive discriminator augmentation.",
    )
    parser.add_argument(
        "--ada-target",
        type=float,
        default=0.6,
        help="Target fraction of positive real discriminator scores for ADA.",
    )
    parser.add_argument(
        "--ada-kimg",
        type=float,
        default=250.0,
        help="Adaptation speed in thousands of images (default: 250 kimg).",
    )
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs")
    parser.add_argument("--weights-dir", type=Path, default=project_root / "weights")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint created by this script.",
    )
    return parser.parse_args()


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, beta: float
) -> None:
    """Update EMA parameters and copy non-parameter buffers."""
    for ema_parameter, parameter in zip(
        ema_model.parameters(), model.parameters(), strict=True
    ):
        ema_parameter.lerp_(parameter, 1.0 - beta)

    for ema_buffer, buffer in zip(
        ema_model.buffers(), model.buffers(), strict=True
    ):
        ema_buffer.copy_(buffer)


@torch.inference_mode()
def save_sample_grid(
    generator: StyleGANGenerator,
    noise: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
) -> None:
    """Save 16 fixed-noise samples per class in two rows of eight."""
    generator.eval()
    images = generator(noise, labels, mix_prob=0.0)
    save_image(
        images,
        output_path,
        nrow=SAMPLES_PER_ROW,
        padding=2,
        normalize=True,
        value_range=(-1, 1),
    )

def plot_loss_curves(loss_d_logs, loss_g_logs, output_dir) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(loss_d_logs, label='Discriminator Loss')
    plt.plot(loss_g_logs, label='Generator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid()
    plt.savefig(output_dir / 'loss_curves.png')
    plt.close()


def save_checkpoint(
    path: Path,
    epoch: int,
    generator: StyleGANGenerator,
    generator_ema: StyleGANGenerator,
    discriminator: StyleGANDiscriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    loss_d_logs: list[float],
    loss_g_logs: list[float],
    ada_p: float,
) -> None:
    """Save model, optimizer, and loss history so training can be resumed."""
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "generator_ema": generator_ema.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "loss_d_logs": loss_d_logs,
            "loss_g_logs": loss_g_logs,
            "ada_p": ada_p,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be zero or greater.")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be at least 1.")
    if args.r1_gamma < 0:
        raise ValueError("--r1-gamma must be zero or greater.")
    if args.r1_interval < 1:
        raise ValueError("--r1-interval must be at least 1.")
    if not 0.0 <= args.ema_beta < 1.0:
        raise ValueError("--ema-beta must be in the range [0, 1).")
    if not 0.0 <= args.mix_prob <= 1.0:
        raise ValueError("--mix-prob must be in the range [0, 1].")
    if not 0.0 < args.ada_target < 1.0:
        raise ValueError("--ada-target must be in the range (0, 1).")
    if args.ada_kimg <= 0:
        raise ValueError("--ada-kimg must be positive.")
    if args.lr_d <= 0 or args.lr_g <= 0:
        raise ValueError("Learning rates must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.weights_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_afhqv2_loaders(
        batch_size=args.batch_size,
        data_dir=str(args.data_dir),
        num_workers=args.num_workers,
        random_horizontal_flip=not args.ada,
    )
    if len(train_loader.dataset) == 0:
        raise ValueError("The training dataset is empty.")
    if tuple(train_loader.dataset.classes) != ("cat", "dog", "wild"):
        raise ValueError(
            "Expected ImageFolder class order ('cat', 'dog', 'wild'), "
            f"got {tuple(train_loader.dataset.classes)}."
        )

    generator = StyleGANGenerator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(
        device
    )
    discriminator = StyleGANDiscriminator().to(device)
    generator_ema = copy.deepcopy(generator).eval()
    generator_ema.requires_grad_(False)

    optimiser_d = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr_d, betas=(0.0, 0.99)
    )
    optimiser_g = torch.optim.Adam(
        generator.parameters(), lr=args.lr_g, betas=(0.0, 0.99)
    )

    loss_d_logs = []
    loss_g_logs = []
    start_epoch = 0
    ada_p = 0.01
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        generator.load_state_dict(checkpoint["generator"])
        generator_ema.load_state_dict(checkpoint["generator_ema"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimiser_g.load_state_dict(checkpoint["optimizer_g"])
        optimiser_d.load_state_dict(checkpoint["optimizer_d"])
        for param_group in optimiser_g.param_groups:
            param_group["lr"] = args.lr_g
        for param_group in optimiser_d.param_groups:
            param_group["lr"] = args.lr_d
        loss_d_logs = checkpoint.get("loss_d_logs", [])
        loss_g_logs = checkpoint.get("loss_g_logs", [])
        ada_p = min(0.8, max(0.01, float(checkpoint.get("ada_p", 0.01))))
        start_epoch = int(checkpoint["epoch"])
        if start_epoch >= args.epochs:
            raise ValueError(
                f"Checkpoint is at epoch {start_epoch}; --epochs must be greater "
                "than the checkpoint epoch."
            )
        print(f"Resuming at epoch {start_epoch + 1} from {args.resume}")

    total_parameters = sum(
        parameter.numel()
        for model in (generator, discriminator)
        for parameter in model.parameters()
    )
    print(f"Generator + discriminator parameters: {total_parameters:,}")

    fixed_labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(
        SAMPLES_PER_CLASS
    )
    fixed_noise = torch.randn(SAMPLES_PER_CLASS, LATENT_DIM, device=device).repeat(NUM_CLASSES, 1)

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()
        interval_real_correct = 0
        interval_real_count = 0
        interval_real_positive = 0
        interval_fake_correct = 0
        interval_fake_count = 0
        interval_loss_d = 0.0
        interval_loss_g = 0.0
        interval_sample_count = 0
        interval_r1_penalty = 0.0
        interval_r1_loss = 0.0
        interval_r1_sample_count = 0

        for batch_index, (real_images, real_labels) in enumerate(
            train_loader, start=1
        ):
            real_images = real_images.to(device, non_blocking=True)
            apply_r1 = (
                args.r1_gamma > 0 and batch_index % args.r1_interval == 0
            )
            real_images = real_images.detach().requires_grad_(apply_r1)
            real_labels = real_labels.to(device, dtype=torch.long, non_blocking=True)
            batch_size = real_images.shape[0]
            expected_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
            if tuple(real_images.shape[1:]) != expected_shape:
                raise ValueError(
                    f"Expected real images shaped [batch, {expected_shape}], "
                    f"got {tuple(real_images.shape)}."
                )

            # Train D to score real images high and generated images low.
            fake_labels = torch.randint(NUM_CLASSES, (batch_size,), device=device)
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            with torch.no_grad():
                fake_images = generator(noise, fake_labels, mix_prob=args.mix_prob)

            optimiser_d.zero_grad(set_to_none=True)
            real_discriminator_images = (
                ada_augment(real_images, ada_p) if args.ada else real_images
            )
            fake_discriminator_images = (
                ada_augment(fake_images, ada_p) if args.ada else fake_images
            )
            real_scores = discriminator(real_discriminator_images, real_labels)
            fake_scores = discriminator(fake_discriminator_images, fake_labels)

            interval_real_correct += (real_scores >= 0.0).sum().item()
            interval_real_count += real_scores.numel()
            interval_real_positive += (real_scores > 0.0).sum().item()
            interval_fake_correct += (fake_scores < 0.0).sum().item()
            interval_fake_count += fake_scores.numel()

            loss_d = F.softplus(-real_scores).mean() + F.softplus(fake_scores).mean()
            if apply_r1:
                real_gradients = torch.autograd.grad(
                    outputs=real_scores.sum(),
                    inputs=real_images,
                    create_graph=True,
                )[0]
                r1_penalty = real_gradients.square().flatten(1).sum(1).mean()
                r1_loss = (
                    0.5 * args.r1_gamma * args.r1_interval * r1_penalty
                )
                loss_d = loss_d + r1_loss
                interval_r1_penalty += r1_penalty.item() * batch_size
                interval_r1_loss += r1_loss.item() * batch_size
                interval_r1_sample_count += batch_size
            loss_d.backward()
            optimiser_d.step()
            optimiser_d.zero_grad(set_to_none=True)

            # Train G to make the discriminator score generated images high.
            generator_labels = torch.randint(
                NUM_CLASSES, (batch_size,), device=device
            )
            generator_noise = torch.randn(batch_size, LATENT_DIM, device=device)
            optimiser_g.zero_grad(set_to_none=True)
            discriminator.requires_grad_(False)
            try:
                generated_images = generator(
                    generator_noise, generator_labels, mix_prob=args.mix_prob
                )
                generator_discriminator_images = (
                    ada_augment(generated_images, ada_p)
                    if args.ada
                    else generated_images
                )
                generated_scores = discriminator(
                    generator_discriminator_images, generator_labels
                )
                loss_g = F.softplus(-generated_scores).mean()
                loss_g.backward()
                optimiser_g.step()
            finally:
                discriminator.requires_grad_(True)

            update_ema(generator_ema, generator, args.ema_beta)
            interval_loss_d += loss_d.item() * batch_size
            interval_loss_g += loss_g.item() * batch_size
            interval_sample_count += batch_size

            if (
                batch_index % args.log_interval == 0
                or batch_index == len(train_loader)
            ):
                success_rate_real = interval_real_correct / interval_real_count
                success_rate_fake = interval_fake_correct / interval_fake_count
                average_loss_d = interval_loss_d / interval_sample_count
                average_loss_g = interval_loss_g / interval_sample_count
                if interval_r1_sample_count:
                    average_r1_penalty = (
                        interval_r1_penalty / interval_r1_sample_count
                    )
                    average_r1_loss = (
                        interval_r1_loss / interval_sample_count
                    )
                    r1_status = (
                        f", R1 penalty: {average_r1_penalty:.2e}, "
                        f"R1 contribution: {average_r1_loss:.2e}"
                    )
                else:
                    r1_status = ""
                real_positive_rate = interval_real_positive / interval_real_count
                if args.ada:
                    adjustment = interval_sample_count / (
                        args.ada_kimg * 1000.0
                    )
                    if real_positive_rate > args.ada_target:
                        ada_p = min(0.8, ada_p + adjustment)
                    elif real_positive_rate < args.ada_target:
                        ada_p = max(0.01, ada_p - adjustment)
                    ada_status = f", ADA p: {ada_p:.3f}"
                else:
                    ada_status = f", ADA p: {ada_p:.3f} (disabled)"
                prefix = f"Epoch [{epoch + 1}/{args.epochs}], Batch [{batch_index}/{len(train_loader)}], "
                print(
                    f"{prefix}Loss_D: {average_loss_d:.4f}, "
                    f"Loss_G: {average_loss_g:.4f}\n"
                    f"{' ' * len(prefix)}D success real: {success_rate_real:.1%}, "
                    f"D success fake: {success_rate_fake:.1%}, "
                    f"D success total: {(success_rate_real + success_rate_fake) / 2:.1%}"
                    f"{r1_status}{ada_status}"
                )
                loss_d_logs.append(average_loss_d)
                loss_g_logs.append(average_loss_g)
                interval_real_correct = 0
                interval_real_count = 0
                interval_real_positive = 0
                interval_fake_correct = 0
                interval_fake_count = 0
                interval_loss_d = 0.0
                interval_loss_g = 0.0
                interval_sample_count = 0
                interval_r1_penalty = 0.0
                interval_r1_loss = 0.0
                interval_r1_sample_count = 0

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] complete."
        )

        sample_path = args.output_dir / f"epoch_{epoch + 1:03d}.png"
        save_sample_grid(generator_ema, fixed_noise, fixed_labels, sample_path)
        print(f"Saved EMA sample grid to {sample_path}")
        plot_loss_curves(loss_d_logs, loss_g_logs, args.output_dir)
        print(f"Saved loss curves to {args.output_dir / 'loss_curves.png'}")

        checkpoint_path = args.weights_dir / "latest_checkpoint.pth"
        save_checkpoint(
            checkpoint_path,
            epoch + 1,
            generator,
            generator_ema,
            discriminator,
            optimiser_g,
            optimiser_d,
            loss_d_logs,
            loss_g_logs,
            ada_p,
        )

    torch.save(generator.state_dict(), args.weights_dir / "generator.pth")
    torch.save(generator_ema.state_dict(), args.weights_dir / "generator_ema.pth")
    torch.save(discriminator.state_dict(), args.weights_dir / "discriminator.pth")
    print(f"Saved final weights and checkpoint in {args.weights_dir}")


if __name__ == "__main__":
    train(parse_args())