"""
ddpm.py: MNIST DDPM from scratch in PyTorch.

This is modeled after the Kaggle notebook "DDPM from scratch in Pytorch":
- forward diffusion q(x_t | x_0)
- reverse DDPM step p(x_{t-1} | x_t)
- sinusoidal timestep embeddings
- a small U-Net with residual convolution blocks and self-attention
- MSE training target: predict the Gaussian noise added to the image

The notebook saves only the best whole model object. This script instead saves
state_dict checkpoints with optimizer and RNG state, so training can resume.
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.utils import save_image


DATA_DIR = Path("learn-pytorch") / "data"
CHECKPOINT_PATH = Path("checkpoints") / "ddpm_mnist.pt"
SAMPLES_DIR = Path("samples")


def get_device():
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_dataloader(data_dir, batch_size, num_workers, device):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        drop_last=True,
    )


class DiffusionForwardProcess:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def add_noise(self, original, noise, timesteps):
        sqrt_alpha_bar_t = self.sqrt_alpha_bars.to(original.device)[timesteps]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars.to(original.device)[timesteps]
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]
        return sqrt_alpha_bar_t * original + sqrt_one_minus_alpha_bar_t * noise


class DiffusionReverseProcess:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample_prev_timestep(self, x_t, noise_pred, timestep):
        device = x_t.device
        beta_t = self.betas.to(device)[timestep]
        alpha_t = self.alphas.to(device)[timestep]
        alpha_bar_t = self.alpha_bars.to(device)[timestep]

        x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        x0 = torch.clamp(x0, -1.0, 1.0)

        mean = (x_t - ((1.0 - alpha_t) * noise_pred) / torch.sqrt(1.0 - alpha_bar_t))
        mean = mean / torch.sqrt(alpha_t)

        if timestep == 0:
            return mean, x0

        alpha_bar_prev = self.alpha_bars.to(device)[timestep - 1]
        variance = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        variance = variance * beta_t
        sigma = torch.sqrt(variance)
        z = torch.randn_like(x_t)
        return mean + sigma * z, x0


def get_time_embedding(time_steps, t_emb_dim):
    if t_emb_dim % 2 != 0:
        raise ValueError("time embedding dimension must be divisible by 2")
    factor = 2 * torch.arange(
        start=0,
        end=t_emb_dim // 2,
        dtype=torch.float32,
        device=time_steps.device,
    )
    factor = 10000 ** (factor / t_emb_dim)
    time_steps = time_steps[:, None]
    t_emb = time_steps / factor
    return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)


def choose_groups(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def choose_heads(channels, max_heads=4):
    heads = min(max_heads, channels)
    while channels % heads != 0:
        heads -= 1
    return heads


class NormActConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm=True, act=True):
        super().__init__()
        self.g_norm = (
            nn.GroupNorm(choose_groups(in_channels), in_channels) if norm else nn.Identity()
        )
        self.act = nn.SiLU() if act else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, n_out, t_emb_dim=128):
        super().__init__()
        self.te_block = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, n_out))

    def forward(self, x):
        return self.te_block(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, num_channels, norm=True):
        super().__init__()
        self.g_norm = (
            nn.GroupNorm(choose_groups(num_channels), num_channels) if norm else nn.Identity()
        )
        self.attn = nn.MultiheadAttention(
            num_channels,
            choose_heads(num_channels),
            batch_first=True,
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, channels, height * width)
        x = self.g_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, k=2, use_conv=True, use_mpool=True):
        super().__init__()
        self.use_conv = use_conv
        self.use_mpool = use_mpool
        self.cv = (
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Conv2d(
                    in_channels,
                    out_channels // 2 if use_mpool else out_channels,
                    kernel_size=4,
                    stride=k,
                    padding=1,
                ),
            )
            if use_conv
            else nn.Identity()
        )
        self.mpool = (
            nn.Sequential(
                nn.MaxPool2d(k, k),
                nn.Conv2d(
                    in_channels,
                    out_channels // 2 if use_conv else out_channels,
                    kernel_size=1,
                ),
            )
            if use_mpool
            else nn.Identity()
        )

    def forward(self, x):
        if not self.use_conv:
            return self.mpool(x)
        if not self.use_mpool:
            return self.cv(x)
        return torch.cat([self.cv(x), self.mpool(x)], dim=1)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, k=2, use_conv=True, use_upsample=True):
        super().__init__()
        self.use_conv = use_conv
        self.use_upsample = use_upsample
        self.cv = (
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels // 2 if use_upsample else out_channels,
                    kernel_size=4,
                    stride=k,
                    padding=1,
                ),
                nn.Conv2d(
                    out_channels // 2 if use_upsample else out_channels,
                    out_channels // 2 if use_upsample else out_channels,
                    kernel_size=1,
                ),
            )
            if use_conv
            else nn.Identity()
        )
        self.up = (
            nn.Sequential(
                nn.Upsample(scale_factor=k, mode="bilinear", align_corners=False),
                nn.Conv2d(
                    in_channels,
                    out_channels // 2 if use_conv else out_channels,
                    kernel_size=1,
                ),
            )
            if use_upsample
            else nn.Identity()
        )

    def forward(self, x):
        if not self.use_conv:
            return self.up(x)
        if not self.use_upsample:
            return self.cv(x)
        return torch.cat([self.cv(x), self.up(x)], dim=1)


class DownC(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=128, num_layers=2, down_sample=True):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList(
            [
                NormActConv(in_channels if i == 0 else out_channels, out_channels)
                for i in range(num_layers)
            ]
        )
        self.conv2 = nn.ModuleList([NormActConv(out_channels, out_channels) for _ in range(num_layers)])
        self.te_block = nn.ModuleList([TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)])
        self.attn_block = nn.ModuleList([SelfAttentionBlock(out_channels) for _ in range(num_layers)])
        self.down_block = Downsample(out_channels, out_channels) if down_sample else nn.Identity()
        self.res_block = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

    def forward(self, x, t_emb):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)
            out = out + self.attn_block[i](out)
        return self.down_block(out)


class MidC(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=128, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList(
            [
                NormActConv(in_channels if i == 0 else out_channels, out_channels)
                for i in range(num_layers + 1)
            ]
        )
        self.conv2 = nn.ModuleList(
            [NormActConv(out_channels, out_channels) for _ in range(num_layers + 1)]
        )
        self.te_block = nn.ModuleList(
            [TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers + 1)]
        )
        self.attn_block = nn.ModuleList([SelfAttentionBlock(out_channels) for _ in range(num_layers)])
        self.res_block = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x, t_emb):
        out = x
        resnet_input = out
        out = self.conv1[0](out)
        out = out + self.te_block[0](t_emb)[:, :, None, None]
        out = self.conv2[0](out)
        out = out + self.res_block[0](resnet_input)

        for i in range(self.num_layers):
            out = out + self.attn_block[i](out)
            resnet_input = out
            out = self.conv1[i + 1](out)
            out = out + self.te_block[i + 1](t_emb)[:, :, None, None]
            out = self.conv2[i + 1](out)
            out = out + self.res_block[i + 1](resnet_input)
        return out


class UpC(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=128, num_layers=2, up_sample=True):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = nn.ModuleList(
            [
                NormActConv(in_channels if i == 0 else out_channels, out_channels)
                for i in range(num_layers)
            ]
        )
        self.conv2 = nn.ModuleList([NormActConv(out_channels, out_channels) for _ in range(num_layers)])
        self.te_block = nn.ModuleList([TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)])
        self.attn_block = nn.ModuleList([SelfAttentionBlock(out_channels) for _ in range(num_layers)])
        self.up_block = Upsample(in_channels, in_channels // 2) if up_sample else nn.Identity()
        self.res_block = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

    def forward(self, x, down_out, t_emb):
        x = self.up_block(x)
        x = torch.cat([x, down_out], dim=1)
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)
            out = out + self.attn_block[i](out)
        return out


class Unet(nn.Module):
    def __init__(
        self,
        im_channels=1,
        base_channels=32,
        t_emb_dim=128,
        num_downc_layers=2,
        num_midc_layers=2,
        num_upc_layers=2,
    ):
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")

        down_ch = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        mid_ch = [base_channels * 8, base_channels * 8, base_channels * 4]
        up_ch = [base_channels * 8, base_channels * 4, base_channels * 2, base_channels // 2]
        down_sample = [True, True, False]

        self.t_emb_dim = t_emb_dim
        self.cv1 = nn.Conv2d(im_channels, down_ch[0], kernel_size=3, padding=1)
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        self.downs = nn.ModuleList(
            [
                DownC(down_ch[i], down_ch[i + 1], t_emb_dim, num_downc_layers, down_sample[i])
                for i in range(len(down_ch) - 1)
            ]
        )
        self.mids = nn.ModuleList(
            [MidC(mid_ch[i], mid_ch[i + 1], t_emb_dim, num_midc_layers) for i in range(len(mid_ch) - 1)]
        )
        up_sample = list(reversed(down_sample))
        self.ups = nn.ModuleList(
            [
                UpC(up_ch[i], up_ch[i + 1], t_emb_dim, num_upc_layers, up_sample[i])
                for i in range(len(up_ch) - 1)
            ]
        )
        self.cv2 = nn.Sequential(
            nn.GroupNorm(choose_groups(up_ch[-1]), up_ch[-1]),
            nn.Conv2d(up_ch[-1], im_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        out = self.cv1(x)
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)

        for mid in self.mids:
            out = mid(out, t_emb)

        for up in self.ups:
            out = up(out, down_outs.pop(), t_emb)

        return self.cv2(out)


def save_checkpoint(path, epoch, global_step, model, optimizer, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": {
            "timesteps": args.timesteps,
            "base_channels": args.base_channels,
            "time_dim": args.time_dim,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        },
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["rng_state"].cpu())
    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_rng_state_all is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    print(f"Resumed checkpoint from {path} at epoch {epoch}, step {global_step}")
    return epoch, global_step


def capture_rng_state():
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng_state(cpu_rng_state, cuda_rng_state_all):
    torch.set_rng_state(cpu_rng_state.cpu())
    if torch.cuda.is_available() and cuda_rng_state_all is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])


def sample_capture_steps(num_timesteps, interval):
    if interval <= 0:
        return []
    return list(range(num_timesteps - interval, 0, -interval))


@torch.no_grad()
def sample_trajectory(
    model,
    reverse_process,
    image_size,
    batch_size,
    device,
    capture_steps,
    channels=1,
):
    model.eval()
    x_t = torch.randn((batch_size, channels, image_size, image_size), device=device)
    frames = {reverse_process.num_timesteps: torch.clamp(x_t.detach().cpu(), -1.0, 1.0)}
    capture_steps = set(capture_steps)
    for timestep in reversed(range(reverse_process.num_timesteps)):
        timestep_label = timestep + 1
        if timestep_label in capture_steps:
            frames[timestep_label] = torch.clamp(x_t.detach().cpu(), -1.0, 1.0)
        t_batch = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
        noise_pred = model(x_t, t_batch)
        x_t, _ = reverse_process.sample_prev_timestep(x_t, noise_pred, timestep)
    frames[0] = torch.clamp(x_t.detach().cpu(), -1.0, 1.0)
    return frames


def save_sample_frames(samples_dir, epoch, frames, nrow):
    epoch_dir = samples_dir / f"epoch-{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    width = max(4, len(str(max(frames))))
    for timestep in sorted(frames.keys(), reverse=True):
        filename = "x0.png" if timestep == 0 else f"t-{timestep:0{width}d}.png"
        save_image((frames[timestep] + 1.0) * 0.5, epoch_dir / filename, nrow=nrow)
    print(f"Saved sample frames to {epoch_dir}")


def save_epoch_samples(model, reverse_process, args, device, epoch):
    cpu_rng_state, cuda_rng_state_all = capture_rng_state()
    try:
        if args.sample_seed >= 0:
            torch.manual_seed(args.sample_seed)
        frames = sample_trajectory(
            model,
            reverse_process,
            args.image_size,
            args.sample_count,
            device,
            sample_capture_steps(args.timesteps, args.sample_interval),
        )
        save_sample_frames(args.samples_dir, epoch, frames, nrow=args.sample_nrow)
    finally:
        restore_rng_state(cpu_rng_state, cuda_rng_state_all)


def train(args):
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using {device} device")

    dataloader = make_dataloader(args.data_dir, args.batch_size, args.num_workers, device)
    model = Unet(base_channels=args.base_channels, t_emb_dim=args.time_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    forward_process = DiffusionForwardProcess(args.timesteps)
    reverse_process = DiffusionReverseProcess(args.timesteps)

    start_epoch = 0
    global_step = 0
    if args.resume:
        if not args.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        start_epoch, global_step = load_checkpoint(args.checkpoint_path, model, optimizer, device)

    sampled_epochs = set()
    if args.sample_every > 0:
        save_epoch_samples(model, reverse_process, args, device, start_epoch)
        sampled_epochs.add(start_epoch)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        for batch_idx, (images, _) in enumerate(dataloader):
            if args.limit_train_batches is not None and batch_idx >= args.limit_train_batches:
                break

            images = images.to(device)
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, args.timesteps, (images.shape[0],), device=device)
            noisy_images = forward_process.add_noise(images, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            noise_pred = model(noisy_images, timesteps)
            loss = criterion(noise_pred, noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            losses.append(loss.item())

            if batch_idx % args.log_every == 0:
                print(
                    f"epoch {epoch + 1:03d}/{args.epochs:03d} "
                    f"batch {batch_idx:04d}/{len(dataloader):04d} "
                    f"loss {loss.item():.4f}"
                )

        avg_loss = sum(losses) / max(1, len(losses))
        print(f"epoch {epoch + 1:03d} done | avg loss {avg_loss:.4f}")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(args.checkpoint_path, epoch + 1, global_step, model, optimizer, args)

        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
            save_epoch_samples(model, reverse_process, args, device, epoch + 1)
            sampled_epochs.add(epoch + 1)

    save_checkpoint(args.checkpoint_path, args.epochs, global_step, model, optimizer, args)
    if args.epochs not in sampled_epochs:
        save_epoch_samples(model, reverse_process, args, device, args.epochs)


def main():
    parser = argparse.ArgumentParser(description="Train a DDPM on MNIST.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--sample-nrow", type=int, default=4)
    parser.add_argument("--sample-interval", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
