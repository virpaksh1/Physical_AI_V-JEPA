
import gc
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

from src.models.vision_transformer import vit_giant


torch.set_grad_enabled(False)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


encoder = vit_giant(
    img_size=224,
    patch_size=16,
)

ckpt = torch.load(
    "checkpoints/vith16.pth.tar",
    map_location="cpu"
)

state_dict = {
    k.replace("module.", ""): v
    for k, v in ckpt["target_encoder"].items()
}

msg = encoder.load_state_dict(
    state_dict,
    strict=False
)

gc.collect()

encoder.eval()

encoder = encoder.half().to(device)

torch.cuda.empty_cache()

print("Encoder loaded")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


frame_paths = sorted(
    glob.glob("frames/*.jpg"))[:60]

frames = []

for fp in frame_paths:

    img = Image.open(fp).convert("RGB")

    tensor = transform(img)

    frames.append(tensor)

frames = torch.stack(frames)

print("Frames tensor:", frames.shape)


fig, axes = plt.subplots(
    2, 4,
    figsize=(12, 6)
)

for ax, fp in zip(
    axes.flatten(),
    frame_paths[:8]
):

    img = Image.open(fp)

    ax.imshow(img)

    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "sample_frames.png"
)

plt.show()


latents = []

with torch.no_grad():

    for i in range(len(frames)):

        x = frames[i].unsqueeze(0)

        x = x.half().to(device)

        feat = encoder(x)

        feat = feat.mean(dim=1)

        latents.append(
            feat.squeeze(0)
            .cpu()
            .numpy()
        )

latents = np.array(latents)

print("Latent shape:", latents.shape)


flow_labels = []
gravity_labels = []

fps = 30
dt = 1 / fps

prev_vy = None

for i in range(len(frame_paths) - 1):

    img1 = cv2.imread(
        frame_paths[i]
    )

    img2 = cv2.imread(
        frame_paths[i + 1]
    )

    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )

    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    vx = flow[..., 0].mean()
    vy = flow[..., 1].mean()

    flow_labels.append([vx, vy])

    if prev_vy is None:
        ay = 0
    else:
        ay = (vy - prev_vy) / dt

    gravity_labels.append([ay])

    prev_vy = vy

flow_labels = np.array(flow_labels)

gravity_labels = np.array(
    gravity_labels
)

latents = latents[:-1]

print("Velocity labels:",
      flow_labels.shape)

print("Gravity labels:",
      gravity_labels.shape)

print("Latents:",
      latents.shape)


X_train, X_test, y_train, y_test = train_test_split(
    latents,
    flow_labels,
    test_size=0.2,
    random_state=42
)

X_train = torch.tensor(
    X_train
).float().to(device)

X_test = torch.tensor(
    X_test
).float().to(device)

y_train = torch.tensor(
    y_train
).float().to(device)

y_test = torch.tensor(
    y_test
).float().to(device)


input_dim = X_train.shape[1]

probe = nn.Linear(
    input_dim,
    2
).to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    probe.parameters(),
    lr=1e-3
)

print("Training velocity probe...")

epochs = 300

losses = []

torch.set_grad_enabled(True)

for epoch in range(epochs):

    probe.train()

    pred = probe(X_train)

    loss = criterion(
        pred,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    losses.append(
        loss.item()
    )

    if epoch % 25 == 0:

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss.item():.6f}"
        )

torch.set_grad_enabled(False)


probe.eval()

with torch.no_grad():

    pred_test = probe(X_test)

velocity_mse = mean_squared_error(
    y_test.cpu().numpy(),
    pred_test.cpu().numpy()
)

print(
    "\nVelocity Test MSE:",
    velocity_mse
)


X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    latents,
    gravity_labels,
    test_size=0.2,
    random_state=42
)

X_train_g = torch.tensor(
    X_train_g
).float().to(device)

X_test_g = torch.tensor(
    X_test_g
).float().to(device)

y_train_g = torch.tensor(
    y_train_g
).float().to(device)

y_test_g = torch.tensor(
    y_test_g
).float().to(device)


gravity_probe = nn.Linear(
    input_dim,
    1
).to(device)

gravity_optimizer = optim.Adam(
    gravity_probe.parameters(),
    lr=1e-3
)

gravity_losses = []

print("Training gravity probe...")

torch.set_grad_enabled(True)

for epoch in range(epochs):

    gravity_probe.train()

    gravity_pred = gravity_probe(
        X_train_g
    )

    gravity_loss = criterion(
        gravity_pred,
        y_train_g
    )

    gravity_optimizer.zero_grad()

    gravity_loss.backward()

    gravity_optimizer.step()

    gravity_losses.append(
        gravity_loss.item()
    )

torch.set_grad_enabled(False)


gravity_probe.eval()

with torch.no_grad():

    gravity_pred_test = gravity_probe(
        X_test_g
    )

gravity_mse = mean_squared_error(
    y_test_g.cpu().numpy(),
    gravity_pred_test.cpu().numpy()
)

print(
    "Gravity Test MSE:",
    gravity_mse
)


plt.figure(figsize=(8, 5))

plt.plot(
    losses,
    label="Velocity Probe"
)

plt.plot(
    gravity_losses,
    label="Gravity Probe"
)

plt.title(
    "Training Loss of Velocity and Gravity Probes"
)

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.legend()

plt.grid(True)

plt.savefig(
    "probe_training_loss.png"
)

plt.show()


pred_np = pred_test.cpu().numpy()

gt_np = y_test.cpu().numpy()

plt.figure(figsize=(8, 8))

plt.scatter(
    gt_np[:, 0],
    pred_np[:, 0],
    label="Horizontal Velocity (vx)"
)

plt.scatter(
    gt_np[:, 1],
    pred_np[:, 1],
    label="Vertical Velocity (vy)"
)

min_val = min(
    gt_np.min(),
    pred_np.min()
)

max_val = max(
    gt_np.max(),
    pred_np.max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "k--",
    label="Ideal Prediction"
)

plt.xlabel(
    "Ground Truth Velocity"
)

plt.ylabel(
    "Predicted Velocity"
)

plt.legend()

plt.title(
    f"Velocity Prediction from V-JEPA Latents\n"
    f"Closer points to diagonal indicate better motion understanding\n"
    f"Test MSE = {velocity_mse:.5f}"
)

plt.grid(True)

plt.savefig(
    "velocity_probe_plot.png"
)

plt.show()


gravity_pred_np = (
    gravity_pred_test
    .cpu()
    .numpy()
)

gravity_gt_np = (
    y_test_g
    .cpu()
    .numpy()
)

plt.figure(figsize=(8, 8))

plt.scatter(
    gravity_gt_np,
    gravity_pred_np,
    alpha=0.7
)

min_val = min(
    gravity_gt_np.min(),
    gravity_pred_np.min()
)

max_val = max(
    gravity_gt_np.max(),
    gravity_pred_np.max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "k--",
    label="Ideal Prediction"
)

plt.xlabel(
    "Ground Truth Vertical Acceleration"
)

plt.ylabel(
    "Predicted Vertical Acceleration"
)

plt.legend()

plt.title(
    f"Gravity-related Motion Prediction\n"
    f"Closer points to diagonal indicate better gravity understanding\n"
    f"Test MSE = {gravity_mse:.5f}"
)

plt.grid(True)

plt.savefig(
    "gravity_probe_plot.png"
)

plt.show()


print("Loading Stable Video Diffusion...")

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16
)

pipe.to(device)

print("SVD loaded")


last_frame = Image.open(
    frame_paths[-1]
).convert("RGB")

last_frame = last_frame.resize(
    (1024, 576)
)

generator = torch.manual_seed(42)

with torch.no_grad():

    video_frames = pipe(
        last_frame,
        decode_chunk_size=8,
        generator=generator,
        motion_bucket_id=127,
        noise_aug_strength=0.02
    ).frames[0]

print(
    "Generated frames:",
    len(video_frames)
)


export_to_video(
    video_frames,
    "predicted_future.mp4",
    fps=8
)

print(
    "Saved: predicted_future.mp4"
)

print("\nSaved files:")
print("1. sample_frames.png")
print("2. probe_training_loss.png")
print("3. velocity_probe_plot.png")
print("4. gravity_probe_plot.png")
print("5. predicted_future.mp4")


"""
LOW VELOCITY TEST MSE
    =>
V-JEPA latent contains motion information

GOOD vx/vy prediction
    =>
latent learned velocity structure

LOW GRAVITY TEST MSE
    =>
latent contains gravity-related motion

GOOD gravity prediction
    =>
latent learned acceleration structure

BAD prediction
    =>
motion information not strongly encoded
"""
