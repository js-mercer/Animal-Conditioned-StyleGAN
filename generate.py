import argparse
import torch
from torchvision.utils import save_image
from pathlib import Path

from models.generator.stylegan_generator import StyleGANGenerator

def parse_args() -> argparse.Namespace:
    """Parse options for generating class-conditioned animal images."""
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate class-conditioned images using a trained AFHQ StyleGAN."
    )
    
    # Parameters
    parser.add_argument("--base_class", type=int, default=0, help="Base class for image generation (0-cat, 1-dog, 2-wild)")
    parser.add_argument("--secondary_class", type=int, default=1, help="Second class for interpolation")
    parser.add_argument("--blend", type=float, default=0.0, help="Proportion of second class embedding (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=4, help="Random seed for reproducibility")
    parser.add_argument("--total_images", type=int, default=1, help="Total number of images to generate (1-64)")
    parser.add_argument("--transition", action="store_true", help="Generates a transition between base and secondary class (overrides --blend)")
    
    # Weight/output paths
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root/"outputs"/"generate",
        help="Directory to save image",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=project_root / "weights" / "generator_ema.pth",
        help="Path to EMA generator weights",
    )

    args = parser.parse_args()
    if args.base_class < 0 or args.base_class > 2 or args.secondary_class < 0 or args.secondary_class > 2:
        parser.error("--base_class and --secondary_class must be between 0 and 2")
    if args.blend < 0 or args.blend >= 1.0 :
            parser.error("--blend must be between 0.0 (inclusive) and 1.0 (exclusive)")
    if args.total_images < 1 or args.total_images > 64:
        parser.error("--total_images must be between 1 and 64")
    if args.transition and args.blend != 0.0:
        parser.error("--transition cannot be used with --blend")

    return args


def generate_animal_image(args):
    # Set random seed for reproducible style features
    torch.manual_seed(args.seed)

    # Ensure output directory exists
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the trained generator and its learned class embeddings.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = StyleGANGenerator().to(device)
    state_dict = torch.load(args.weights_dir, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    embeddings = generator.mapping.label_embed.weight.detach()

    if args.transition:
        # Reuse one latent vector so the class change is the main difference per image.
        fixed_noise = torch.randn(1, 512, device=device).expand(10, -1)
        alphas = torch.linspace(0, 1, 10, device=device).unsqueeze(1)
        labels = (
            (1 - alphas) * embeddings[args.base_class]
            + alphas * embeddings[args.secondary_class]
        )

        with torch.no_grad():
            generated_images = generator(fixed_noise, labels, mix_prob=0.0)

        # Save the generated transition images in a grid
        save_image(
            generated_images, 
            output_dir / f"generated_image(s).png", 
            nrow=10,
            padding=2,
            normalize=True,
            value_range=(-1, 1),
            )

    else:
        # Give every image distinct noise while keeping its class embedding fixed.
        noise = torch.randn(args.total_images, 512, device=device)
        label = (
            (1 - args.blend) * embeddings[args.base_class]
            + args.blend * embeddings[args.secondary_class]
        ).unsqueeze(0).expand(args.total_images, -1)

        with torch.no_grad():
            generated_images = generator(noise, label, mix_prob=0.0)

        # Save each generated image in a grid
        save_image(
            generated_images, 
            output_dir / f"generated_image(s).png", 
            padding=2,
            normalize=True,
            value_range=(-1, 1),
            )

    print(f"Generated image(s) saved to: {output_dir / f'generated_image(s).png'}")

if __name__ == "__main__":
    arguments = parse_args()
    generate_animal_image(arguments)