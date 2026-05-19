
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

from src.models.vision_transformer import vit_large



torch.set_grad_enabled(False)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")




encoder = vit_large(
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

#print(msg)

gc.collect()

encoder.eval()

# FP16 for GPU memory savings
encoder = encoder.half().to(device)

torch.cuda.empty_cache()

print("Encoder loaded")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])



frame_paths = sorted(glob.glob("frames/*.jpg"))[:62]

frames = []

for fp in frame_paths:

    img = Image.open(fp).convert("RGB")

    tensor = transform(img)

    frames.append(tensor)

# [T,C,H,W]
frames = torch.stack(frames)

print("Frames tensor:", frames.shape)



fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for ax, fp in zip(axes.flatten(), frame_paths[:8]):

    img = Image.open(fp)

    ax.imshow(img)

    ax.axis("off")

plt.tight_layout()
plt.show()




latents = []

with torch.no_grad():

    for i in range(len(frames)):

        # [1,C,H,W]
        x = frames[i].unsqueeze(0)

        # IMPORTANT:
        # match encoder dtype (FP16)
        x = x.half().to(device)

        feat = encoder(x)

        # global average pooling
        feat = feat.mean(dim=1)

        latents.append(
            feat.squeeze(0).cpu().numpy()
        )

latents = np.array(latents)

print("Latent shape:", latents.shape)



flow_labels = []

for i in range(len(frame_paths) - 1):

    img1 = cv2.imread(frame_paths[i])
    img2 = cv2.imread(frame_paths[i + 1])

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

flow_labels = np.array(flow_labels)

# remove last latent to align lengths
latents = latents[:-1]

print("Flow labels:", flow_labels.shape)
print("Latents:", latents.shape)




X_train, X_test, y_train, y_test = train_test_split(
    latents,
    flow_labels,
    test_size=0.2,
    random_state=42
)

X_train = torch.tensor(X_train).float().to(device)
X_test  = torch.tensor(X_test).float().to(device)

y_train = torch.tensor(y_train).float().to(device)
y_test  = torch.tensor(y_test).float().to(device)




input_dim = X_train.shape[1]

probe = nn.Linear(input_dim, 2).to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    probe.parameters(),
    lr=1e-3
)

print("Training linear probe...")



epochs = 300

losses = []

torch.set_grad_enabled(True)

for epoch in range(epochs):

    probe.train()

    pred = probe(X_train)

    loss = criterion(pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

    if epoch % 25 == 0:

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss.item():.6f}"
        )

torch.set_grad_enabled(False)




probe.eval()

with torch.no_grad():

    pred_test = probe(X_test)

mse = mean_squared_error(
    y_test.cpu().numpy(),
    pred_test.cpu().numpy()
)

print("\nTest MSE:", mse)




plt.figure(figsize=(8, 5))

plt.plot(losses)

plt.title("Linear Probe Training Loss")

plt.xlabel("Epoch")

plt.ylabel("MSE")

plt.grid(True)

plt.show()




pred_np = pred_test.cpu().numpy()

gt_np = y_test.cpu().numpy()

plt.figure(figsize=(8, 8))

plt.scatter(
    gt_np[:, 0],
    pred_np[:, 0],
    label="vx"
)

plt.scatter(
    gt_np[:, 1],
    pred_np[:, 1],
    label="vy"
)

plt.xlabel("Ground Truth")

plt.ylabel("Predicted")

plt.legend()

plt.title("Velocity Probe")

plt.grid(True)

plt.savefig('velocity_probe_plot.png')
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

last_frame = last_frame.resize((1024, 576))

generator = torch.manual_seed(42)

with torch.no_grad():

    video_frames = pipe(
        last_frame,
        decode_chunk_size=8,
        generator=generator,
        motion_bucket_id=127,
        noise_aug_strength=0.02
    ).frames[0]

print("Generated frames:", len(video_frames))




export_to_video(
    video_frames,
    "predicted_future.mp4",
    fps=8
)

print("Saved: predicted_future.mp4")




"""
LOW TEST MSE
    =>
V-JEPA latent contains motion information

GOOD vx/vy prediction
    =>
latent learned velocity structure

BAD prediction
    =>
motion info not strongly encoded
"""

