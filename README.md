# Physical_AI_V-JEPA

# V-JEPA Motion Understanding Probe

This project evaluates whether **V-JEPA latent representations encode motion information**, specifically **velocity dynamics**, using a simple **linear probing approach**.

The pipeline extracts latent embeddings from a video using a **V-JEPA ViT-Large encoder**, computes **optical flow-based motion labels**, and trains lightweight probes to predict motion properties from frozen latent representations.

Future work includes extending the analysis toward **gravity-related motion understanding**.

---

## Overview

The workflow consists of:

1. **Load V-JEPA ViT-Large encoder**
2. **Extract latent embeddings from video frames**
3. **Compute optical flow velocities (vx, vy)**
4. **Train a linear probe on frozen latent space**
5. **Evaluate motion understanding**
6. **Generate future video using Stable Video Diffusion**

---

## Project Structure

```text
project/
│── run.py                         # Main script to run the complete pipeline
│── frames/                        # Input video frames (.jpg)
│── checkpoints/
│   └── vith16.pth.tar            # V-JEPA checkpoint
│── src/
│   └── models/
│       └── vision_transformer.py
│── outputs/
│   ├── sample_frames.png
│   ├── probe_training_loss.png
│   ├── velocity_probe_plot.png
│   ├── gravity_probe_plot.png
│   └── predicted_future.mp4
│── README.md
```

---

## Model Used

### V-JEPA Encoder
- **Backbone:** ViT-Large
- **Checkpoint:** `vith16.pth.tar`
- **Input Size:** `224 × 224`
- **Patch Size:** `16`

Latent representations are extracted from video frames and used to evaluate whether motion information exists in the learned representation space.

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib numpy pillow scikit-learn
pip install diffusers transformers accelerate
```

---

## Dataset Preparation

Extract frames from a video and place them inside the `frames/` folder.

Example:

```text
frames/
├── frame_0001.jpg
├── frame_0002.jpg
├── frame_0003.jpg
...
```

The script reads the first **60 frames**.

---

## Running the Code

Run:

```bash
python run.py
```

This executes the full pipeline:

- Loads V-JEPA encoder
- Extracts latent representations
- Computes optical flow velocity labels
- Trains velocity and gravity probes
- Generates evaluation plots
- Predicts future video using Stable Video Diffusion

---

## Methodology

### 1. Latent Extraction

Frames are passed through a frozen **V-JEPA ViT-Large encoder**.

The encoder output is mean pooled to obtain a compact latent representation.

```python
feat = encoder(x)
feat = feat.mean(dim=1)
```

---

### 2. Velocity Label Generation

Motion labels are computed using **Farneback Optical Flow**.

Velocity components:

- **vx** → horizontal motion
- **vy** → vertical motion

```python
flow = cv2.calcOpticalFlowFarneback(...)
vx = flow[..., 0].mean()
vy = flow[..., 1].mean()
```

---

### 3. Gravity Approximation

Vertical acceleration is approximated using:

:contentReference[oaicite:0]{index=0}

This acts as a proxy for gravity-related motion.

---

### 4. Linear Probe

A simple **linear layer** is trained on frozen latent embeddings.

Velocity probe:

```python
probe = nn.Linear(input_dim, 2)
```


## Results

### Velocity Understanding
- V-JEPA latent space shows **partial motion understanding**
- Velocity information (**vx, vy**) is encoded to some extent


## Future Work

- Improve motion probing with temporal transformers
- Add gravity probe
- Compare **V-JEPA vs I-JEPA**
- Use longer video sequences
- Investigate physics-aware latent representations

---

## Technologies Used

- PyTorch
- V-JEPA (ViT-Large)
- OpenCV Optical Flow
- Stable Video Diffusion
- Scikit-learn
- Matplotlib
```
