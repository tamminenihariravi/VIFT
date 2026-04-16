# VIFT_DUMMY — Visual-Inertial Fusion Transformer for Odometry

## Complete Technical Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [System Architecture](#3-system-architecture)
4. [Two-Stage Pipeline](#4-two-stage-pipeline)
5. [Stage 1 — Feature Encoding & Caching](#5-stage-1--feature-encoding--caching)
6. [Stage 2 — Pose Transformer Training](#6-stage-2--pose-transformer-training)
7. [Model Architectures (Detailed)](#7-model-architectures-detailed)
8. [Loss Functions & Training Theory](#8-loss-functions--training-theory)
9. [Data Pipeline](#9-data-pipeline)
10. [Evaluation & Metrics](#10-evaluation--metrics)
11. [Configuration System (Hydra)](#11-configuration-system-hydra)
12. [File Reference](#12-file-reference)
13. [Usage Commands](#13-usage-commands)
14. [Suggestions to Improve Accuracy](#14-suggestions-to-improve-accuracy)

---

## 1. Project Overview

**VIFT_DUMMY** is a lightweight, edge-deployable **Visual-Inertial Odometry (VIO)** system that estimates 6-DoF (Degrees of Freedom) camera pose from monocular image sequences and IMU (Inertial Measurement Unit) sensor data. It is the experimental/development branch of the larger VIFT project, designed for rapid iteration on the **KITTI Odometry Benchmark**.

### Core Design Philosophy

The project decouples the expensive feature extraction from the lightweight pose estimation:

1. **Stage 1 (Offline):** A heavy, pre-trained CNN encoder extracts 768-dimensional latent features from raw images and IMU data, caching them as `.npy` files.
2. **Stage 2 (Trainable):** A compact Transformer-based model consumes these cached latent features to predict frame-to-frame 6-DoF relative poses.

This two-stage architecture enables:
- Training the pose head on machines with **≤ 4 GB VRAM**
- Sub-second inference latency for real-time applications
- Ablation studies (visual-only, inertial-only, fused) without re-running the encoder

---

## 2. Problem Statement & Motivation

### What is Visual-Inertial Odometry?

Visual-Inertial Odometry estimates the trajectory of a moving platform (car, robot, drone) by fusing:
- **Visual data:** Sequential camera images capturing the environment
- **Inertial data:** IMU readings (accelerometer + gyroscope) measuring linear acceleration and angular velocity

The goal is to predict the **relative pose** between consecutive frames, represented as 6 values:

$$
\text{pose} = [\underbrace{\theta_x, \theta_y, \theta_z}_{\text{rotation (Euler angles)}}, \underbrace{t_x, t_y, t_z}_{\text{translation}}]
$$

### Why Sensor Fusion?

| Modality | Strengths | Weaknesses |
|----------|-----------|------------|
| **Vision** | Rich spatial information, texture | Fails in dark/featureless scenes |
| **IMU** | Works in any lighting, high frequency | Accumulates drift over time |
| **Fusion** | Combines both strengths | Requires careful alignment |

### Why a Two-Stage Pipeline?

Traditional end-to-end VIO models require enormous GPU memory to process raw images through deep CNNs during training. By caching the encoder output, we:

1. **Eliminate redundant computation** — The encoder runs once per sample; the Transformer trains over hundreds of epochs on the tiny cached vectors.
2. **Enable edge deployment** — Only the lightweight Transformer head needs to run at inference time (when paired with an on-device encoder or pre-computed features).
3. **Accelerate experimentation** — Different Transformer architectures, loss functions, and hyperparameters can be tested without re-encoding.

---

## 3. System Architecture

### High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        VIFT_DUMMY Pipeline                          │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              STAGE 1: Feature Encoding (Offline)               │  │
│  │                                                                │  │
│  │  ┌─────────────┐     ┌──────────────────┐     ┌───────────┐  │  │
│  │  │ KITTI Images ├────►│  Visual Encoder  ├────►│           │  │  │
│  │  │ (256 × 512)  │     │  (6-layer CNN)   │     │   Cat()   │  │  │
│  │  └─────────────┘     │  → 512-d vector  │     │           │  │  │
│  │                      └──────────────────┘     │  → 768-d  ├──┼──► .npy cache
│  │  ┌─────────────┐     ┌──────────────────┐     │   latent  │  │  │
│  │  │  IMU Data   ├────►│ Inertial Encoder ├────►│   vector  │  │  │
│  │  │ (6-axis)     │     │  (3-layer Conv1d)│     │           │  │  │
│  │  └─────────────┘     │  → 256-d vector  │     └───────────┘  │  │
│  │                      └──────────────────┘                     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              STAGE 2: Pose Estimation (Training)               │  │
│  │                                                                │  │
│  │  ┌───────────┐  ┌────────────┐  ┌─────────────┐  ┌─────────┐│  │
│  │  │ .npy cache├─►│ Linear(768 ├─►│ Transformer ├─►│ FC Head ││  │
│  │  │ (768-d)   │  │   → 128)   │  │ Encoder     │  │ (128→6) ││  │
│  │  └───────────┘  │ + PosEmbed │  │ (2 layers,  │  │ 6-DoF   ││  │
│  │                 └────────────┘  │  8 heads)    │  │ output  ││  │
│  │                                 └─────────────┘  └─────────┘│  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
Raw KITTI Data
    │
    ├── Images (PNG, 1241×376) ──► Resize to 256×512
    │                              ──► Pair consecutive frames (6-ch)
    │                              ──► 6-layer CNN ──► 512-d visual feature
    │
    ├── IMU (.mat, 6-axis @ 10Hz) ──► 3-layer Conv1d ──► 256-d inertial feature
    │
    └── Poses (.txt, 12 params)   ──► Relative 6-DoF (Euler + translation)

Cached Latent: [visual(512) ‖ inertial(256)] = 768-d per frame

Pose Transformer: 768-d input → 128-d embedding → Transformer → 6-DoF pose estimates
```

---

## 4. Two-Stage Pipeline

### Stage 1 → `cache_latent_features.py`

| Aspect | Detail |
|--------|--------|
| **Input** | Raw KITTI images + IMU data |
| **Encoder** | Pre-trained CNN (`vf_512_if_256_3e-05.model`) |
| **Output** | `.npy` files with 768-d latent vectors per frame |
| **GPU required** | Yes (for inference only), ~2 GB VRAM |
| **Run once** | Per dataset split (train/val) |

### Stage 2 → `train.py`

| Aspect | Detail |
|--------|--------|
| **Input** | Cached `.npy` latent vectors |
| **Model** | PoseTransformer (~264K parameters) |
| **Output** | 6-DoF relative pose per frame-pair |
| **GPU required** | Yes, but very lightweight (~500 MB VRAM) |
| **Run many times** | For each experiment/hyperparameter variant |

---

## 5. Stage 1 — Feature Encoding & Caching

### Visual Encoder (`OriginalEncoder`)

The visual encoder is a deep CNN that takes paired consecutive frames (6 channels: RGB of frame *t* concatenated with RGB of frame *t+1*) and outputs a 512-dimensional feature vector:

```
Input: (B × seq_len, 6, 256, 512)
    │
    ├── Conv2d(6 → 64, k=7, s=2) + BN + LeakyReLU + Dropout(0.2)
    ├── Conv2d(64 → 128, k=5, s=2) + BN + LeakyReLU + Dropout(0.2)
    ├── Conv2d(128 → 256, k=5, s=2) + BN + LeakyReLU + Dropout(0.2)
    ├── Conv2d(256 → 256, k=3, s=1) + BN + LeakyReLU + Dropout(0.2)
    ├── Conv2d(256 → 512, k=3, s=2) + BN + LeakyReLU + Dropout(0.2)
    ├── Conv2d(512 → 512, k=3, s=1) + BN + LeakyReLU + Dropout(0.2)
    ├── Conv2d(512 → 512, k=3, s=2) + BN + LeakyReLU + Dropout(0.2)
    ├── Conv2d(512 → 512, k=3, s=1) + BN + LeakyReLU + Dropout(0.2)
    ├── Conv2d(512 → 1024, k=3, s=2) + BN + LeakyReLU + Dropout(0.5)
    └── Flatten → Linear(feature_size → 512)
Output: (B, seq_len-1, 512)
```

### Inertial Encoder (`OriginalInertialEncoder`)

Processes windowed IMU readings (6-axis: 3 accelerometer + 3 gyroscope channels) using 1D convolutions:

```
Input: (B × seq_len, 6, 11)   ← 11 IMU readings per frame window
    │
    ├── Conv1d(6 → 64, k=3) + BN + LeakyReLU + Dropout(0.1)
    ├── Conv1d(64 → 128, k=3) + BN + LeakyReLU + Dropout(0.1)
    ├── Conv1d(128 → 256, k=3) + BN + LeakyReLU + Dropout(0.1)
    └── Flatten → Linear(256×11 → 256)
Output: (B, seq_len, 256)
```

### Feature Concatenation & Caching

```python
latent_vector = torch.cat((feat_visual, feat_inertial), dim=2)
# Shape: (seq_len, 768) — 512 visual + 256 inertial
```

Each sample is saved as four `.npy` files:
- `{i}.npy` — Latent vector `(seq_len, 768)`
- `{i}_gt.npy` — Ground-truth relative poses `(seq_len-1, 6)`
- `{i}_rot.npy` — Segment rotation for weighting
- `{i}_w.npy` — Sample weight from LDS

---

## 6. Stage 2 — Pose Transformer Training

### How the Transformer Works for Pose Estimation

The keyinsight is treating sequential frames as "tokens" in a sequence, analogous to words in a sentence for NLP. The Transformer's self-attention mechanism lets each frame attend to all previous frames (via causal masking), capturing temporal motion patterns.

#### Step-by-Step Forward Pass

1. **Dimensionality Reduction:** `Linear(768 → 128)` compresses the cached latent vectors
2. **Positional Encoding:** Sinusoidal embeddings encode temporal order
3. **Causal Self-Attention:** Each frame can only attend to itself and past frames (prevents information leakage from the future)
4. **Feed-Forward Network:** Two-layer MLP inside each Transformer layer
5. **Pose Regression Head:** `Linear(128 → 128) → LeakyReLU → Linear(128 → 6)` outputs 6-DoF pose

### Attention Mechanism (Core Theory)

The self-attention formula from "Attention Is All You Need" (Vaswani et al., 2017):

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Where:
- **Q (Query):** "What am I looking for?" — the current frame's query vector
- **K (Key):** "What information do I have?" — keys from all accessible frames
- **V (Value):** "What is the actual content?" — values to be weighted and summed
- **d_k:** Dimension of keys, used for scaling to prevent gradient vanishing

### Causal Masking

To prevent the model from "cheating" by looking at future frames when predicting the current pose, a **causal mask** is applied:

```
Frame:    1     2     3     4
    1  [ 0.0  -inf  -inf  -inf ]   ← Frame 1 sees only itself
    2  [ 0.0   0.0  -inf  -inf ]   ← Frame 2 sees frames 1,2
    3  [ 0.0   0.0   0.0  -inf ]   ← Frame 3 sees frames 1,2,3
    4  [ 0.0   0.0   0.0   0.0 ]   ← Frame 4 sees all past frames
```

Positions with `-inf` become 0 after softmax, effectively blocking future information.

### Positional Embedding

Since Transformers have no inherent notion of sequence order, sinusoidal positional embeddings are added:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

This gives each frame a unique, bounded position "barcode" that:
- Stays in `[-1, 1]` regardless of sequence length
- Encodes relative distances between frames
- Generalizes to unseen sequence lengths

---

## 7. Model Architectures (Detailed)

### 7.1. VIOSimpleDenseNet (Baseline/Sanity Check)

**Purpose:** A minimal fully-connected network used as a pipeline sanity check — verifying that data flows correctly, loss computes, and gradients update.

```
Input: Images (B, seq_len, C, H, W) + IMU (B, imu_samples, 6)
    │
    ├── Feature Extractor (per frame):
    │   ├── Conv2d(C → 16, k=5, s=2) + BN + ReLU
    │   └── AdaptiveAvgPool2d((4,4))
    │   → (B, seq_len × 16 × 4 × 4)
    │
    ├── Flatten + Concatenate with flattened IMU
    │
    └── MLP: Linear(input → 256) → BN → ReLU
              Linear(256 → 256) → BN → ReLU
              Linear(256 → 256) → BN → ReLU
              Linear(256 → (seq_len-1)×6)
              → Reshape to (B, seq_len-1, 6)
```

### 7.2. PoseTransformer (Primary Model)

**Purpose:** Main production model for visual-inertial pose estimation from cached latent features.

| Hyperparameter | Default Value | Description |
|----------------|---------------|-------------|
| `input_dim` | 768 | Cached latent vector size (512 visual + 256 inertial) |
| `embedding_dim` | 128 | Internal embedding size |
| `num_layers` | 2 | Number of Transformer encoder layers |
| `nhead` | 8 | Number of attention heads |
| `dim_feedforward` | 512 | FFN hidden dimension per layer |
| `dropout` | 0.1 | Dropout rate |

**Architecture:**
```
Latent Input (B, seq_len, 768)
    │
    ├── Linear(768 → 128)
    ├── + Sinusoidal Positional Embedding
    │
    ├── TransformerEncoder (2 layers):
    │   ├── MultiHeadAttention(128, 8 heads) + causal mask
    │   ├── LayerNorm + Residual Connection
    │   ├── FFN: Linear(128 → 512) → ReLU → Linear(512 → 128)
    │   └── LayerNorm + Residual Connection
    │
    └── Pose Head:
        ├── Linear(128 → 128)
        ├── LeakyReLU(0.1)
        └── Linear(128 → 6)  → (B, seq_len, 6)
```

**Parameter Count Estimate:**
- FC1: 768 × 128 = ~98K
- Transformer (2 layers, 8 heads): ~2 × (4 × 128² + 2 × 128 × 512) ≈ ~394K
- FC2: 128 × 128 + 128 × 6 ≈ ~17K
- **Total: ~509K parameters** (extremely lightweight)

### 7.3. PoseTransformerVisual (Ablation — Vision Only)

**Purpose:** Uses only the visual features (first 512 dimensions of the 768-d latent) to measure how much the vision branch alone contributes to accuracy.

```python
visual_inertial_features = visual_inertial_features[:, :, :512]
```

### 7.4. PoseTransformerInertial (Ablation — IMU Only)

**Purpose:** Uses only the inertial features (last 256 dimensions) to measure how much the IMU branch alone contributes.

```python
visual_inertial_features = visual_inertial_features[:, :, 512:]
```

### 7.5. TokenizedPoseTransformer (Advanced — Classification-based)

**Purpose:** Treats pose estimation as a **classification problem** instead of regression. Continuous pose values are discretized into tokens (bins), similar to how language models predict the next word.

**Key Innovation:**
- Uses `OdometryBins` tokenizer to convert continuous pose values into discrete token IDs
- Per-component embeddings (6 separate embedding tables for the 6 DoF)
- Cross-entropy loss instead of MSE/MAE
- Handles rare motion patterns (sharp turns) better than regression

**Architecture:**
```
Latent Input (B, seq_len, 768)
    │
    ├── Linear(768 → 128) + Positional Embedding
    │
    ├── Ground Truth → Tokenize (6 components → 6 token IDs each)
    │   └── 6 × Embedding(n_tokens → 128) → Sum → + Positional Embedding
    │
    ├── Concatenate: [latent_embeddings, token_embeddings]
    │   → (B, 2×seq_len, 128)
    │
    ├── TransformerEncoder with causal mask
    │
    └── 6 × Classification Head:
        Linear(128 → 128) → LeakyReLU → Linear(128 → n_tokens)
        → Argmax → Detokenize → Continuous pose values
```

### Ablation Study Design

The four model variants enable a rigorous **ablation study**:

| Model | Input | Purpose |
|-------|-------|---------|
| `PoseTransformer` | Visual + Inertial (768) | Full fusion performance |
| `PoseTransformerVisual` | Visual only (512) | Vision contribution |
| `PoseTransformerInertial` | Inertial only (256) | IMU contribution |
| `TokenizedPoseTransformer` | Visual + Inertial (768) | Classification vs. regression |

---

## 8. Loss Functions & Training Theory

### 8.1. Scale Imbalance Problem

Rotation angles (radians, typically 0.001–0.01) and translations (meters, typically 0.1–2.0) differ by **2–3 orders of magnitude**. Without proper weighting, the optimizer will ignore rotation entirely.

### 8.2. Loss Functions

#### WeightedMSEPoseLoss
```python
loss = angle_weight × MSE(pred_rot, gt_rot) + MSE(pred_trans, gt_trans)
```
- Default `angle_weight = 100`
- Simple but effective baseline

#### WeightedMAEPoseLoss
```python
loss = angle_weight × L1(pred_rot, gt_rot) + L1(pred_trans, gt_trans)
```
- Default `angle_weight = 10`
- More robust to outliers than MSE

#### RPMGPoseLoss (Riemannian Projected Manifold Gradient)
```python
R_pred = EulerToRotMatrix(pred_rot)           # 3×3 rotation matrix
R_proj = RPMG.apply(R_pred.flatten(), τ, λ)   # Project onto SO(3)
loss = angle_weight × L1(R_proj, R_gt) + L1(pred_trans, gt_trans)
```

**Theory:** Euler angles are problematic for gradient-based optimization because:
- They suffer from **gimbal lock** (singularity at ±90° pitch)
- The loss landscape has **discontinuities** at angle wrapping boundaries

RPMG solves this by:
1. Converting predicted Euler angles to a 3×3 rotation matrix
2. Projecting it onto the **SO(3) manifold** (the space of valid rotations) using SVD
3. Computing the Riemannian gradient (gradient on the curved manifold surface, not in flat Euclidean space)
4. This ensures gradients always point in the direction of valid rotations

#### DataWeightedRPMGPoseLoss
Extends RPMG loss with per-sample importance weighting from Label Distribution Smoothing (LDS).

#### CustomWeightedPoseLoss
Applies manually tuned per-component weights for each of the 6 DoF:
```python
loss = 0.667×L1(θ_x) + 0.2×L1(θ_y) + 1.0×L1(θ_z)
     + 0.1×L1(t_x) + 0.1×L1(t_y) + 0.03×L1(t_z)
```

#### TokenizedPoseLoss
For the `TokenizedPoseTransformer`, uses **cross-entropy loss** on the discretized pose token predictions.

### 8.3. Label Distribution Smoothing (LDS)

The KITTI dataset is heavily **imbalanced**: most frames involve straight driving (small rotation), while sharp turns (large rotation) are rare. This causes the model to over-fit to straight driving and fail on turns.

**LDS Solution Process:**

```
Step 1: Compute rotation angle per sample
Step 2: Bin angles into 10 equal-width bins
Step 3: Count samples per bin (empirical distribution)
Step 4: Smooth with Gaussian kernel (σ=5, size=7) to handle empty bins
Step 5: Weight = 1 / smoothed_count
```

Result: rare rotation angles get **high weights**, common ones get **low weights**, forcing the model to learn equally across the distribution.

Available smoothing kernels:
- **Gaussian:** Bell-curve shaped, smooth dropoff → `▂ ▄ ▆ █ ▆ ▄ ▂`
- **Triangle:** Linear dropoff → `▃ ▅ ▆ █ ▆ ▅ ▃`
- **Laplace:** Sharp, peaked distribution → `▂ ▄ █ ▄ ▂`

---

## 9. Data Pipeline

### 9.1. KITTI Dataset (`KITTI_dataset.py`)

**KITTI Odometry Benchmark:** 22 sequences of driving data with:
- Synchronized stereo camera images (left color: `image_2/`)
- Interpolated IMU data (`.mat` files)
- Ground-truth 4×4 pose matrices (`.txt` files)

**Train/Val Split:**
| Split | Sequences |
|-------|-----------|
| Training | 00, 01, 02, 04, 06, 08, 09 |
| Validation | 05, 07, 10 |

**Sample Construction:**
- A sliding window of `seq_len` frames (default 11) produces one sample
- Each sample contains:
  - `imgs`: Sequence of `seq_len` images
  - `imus`: IMU readings for `(seq_len-1) × 10 + 1` timesteps
  - `gts`: `(seq_len-1)` relative 6-DoF poses
  - `rot`: Segment rotation for LDS weighting
  - `weight`: LDS-derived importance weight

### 9.2. Data Transforms (`custom_transform.py`)

Applied in a composable pipeline:

| Transform | Effect |
|-----------|--------|
| `ToTensor` | PIL → Tensor, zero-centered to `[-0.5, 0.5]` |
| `Resize((256, 512))` | Uniform image dimensions |
| `Normalize(mean, std)` | ImageNet normalization |
| `RandomHorizontalFlip(p=0.5)` | Mirror augmentation (+ IMU/GT sign flip) |
| `RandomColorAug` | Gamma, brightness, color jitter |

### 9.3. Latent Vector Dataset (`latent_kitti_dataset.py`)

After Stage 1 caching, the training loop uses a much simpler dataset:

```python
# Each sample = 4 files
latent_vector = load(f"{idx}.npy")     # (seq_len, 768)
gt            = load(f"{idx}_gt.npy")   # (seq_len, 6)
rot           = load(f"{idx}_rot.npy")  # scalar
weight        = load(f"{idx}_w.npy")    # scalar
```

### 9.4. Lightning DataModule (`vio_datamodule.py`)

Wraps datasets into PyTorch Lightning's `LightningDataModule` interface:
- Configurable `batch_size`, `num_workers`, `pin_memory`
- Separate train/val/test dataloaders
- Integrates with Hydra for configuration injection

---

## 10. Evaluation & Metrics

### 10.1. Pose Accumulation

Relative poses are chained to reconstruct the global trajectory:

```python
def path_accu(relative_poses):
    """Chain relative poses into global trajectory."""
    global_poses = [np.eye(4)]  # Start at origin
    for pose in relative_poses:
        R = euler_to_rotation_matrix(pose[:3])
        t = pose[3:]
        T_rel = build_4x4_matrix(R, t)
        global_poses.append(global_poses[-1] @ T_rel)
    return global_poses
```

### 10.2. KITTI Metrics

The standard KITTI odometry metrics evaluate on sub-trajectories of lengths 100–800 meters:

| Metric | Formula | Unit |
|--------|---------|------|
| **t_rel** | Mean translational error / path length | % |
| **r_rel** | Mean rotational error / path length | deg/100m |
| **t_RMSE** | √(mean(Σ(t_est - t_gt)²)) | meters |
| **r_RMSE** | √(mean(Σ(θ_est - θ_gt)²)) | degrees |

### 10.3. Evaluation Pipeline

Two evaluation modes:

1. **Latent Evaluation (`kitti_latent_eval.py`):**
   - At test time, runs the full pipeline: raw images → encoder → latent → Transformer → pose
   - Requires the pre-trained encoder weights
   - Supports history-based evaluation (sliding window)

2. **Standard Evaluation (`kitti_eval.py`):**
   - For the `VIOSimpleDenseNet` baseline (end-to-end)

### 10.4. Output Artifacts

After evaluation, the system produces:
- 2D trajectory plots (estimated vs. ground truth)
- Speed heatmaps overlaid on the trajectory
- Predicted pose text files (KITTI format)
- Per-sequence error metrics

---

## 11. Configuration System (Hydra)

### Overview

The project uses **Hydra** (by Facebook Research) for hierarchical configuration management. All model, data, training, and experiment parameters are defined in YAML files under `configs/`.

### Directory Structure

```
configs/
├── train.yaml              # Root training config
├── eval.yaml               # Root evaluation config
├── model/
│   ├── vio.yaml            # VIOSimpleDenseNet
│   ├── latent_vio_tf.yaml  # PoseTransformer
│   └── weighted_latent_vio_tf.yaml  # WeightedVIOLitModule + PoseTransformer
├── data/
│   ├── dummy_vio.yaml      # Small-scale sanity check data
│   └── latent_kitti_vio.yaml  # Latent KITTI dataset
├── experiment/
│   ├── dummy_vio.yaml      # Quick pipeline test
│   ├── latent_kitti_vio_tf.yaml  # Standard training
│   ├── latent_kitti_vio_weighted_tf.yaml  # With LDS weighting
│   └── ... (15 experiment variants)
├── trainer/
├── callbacks/
├── logger/
└── hydra/
```

### Key Experiment Configurations

| Experiment | Model | Loss | Features |
|------------|-------|------|----------|
| `latent_kitti_vio_tf` | PoseTransformer | WeightedMAE | Standard training |
| `latent_kitti_vio_weighted_tf` | PoseTransformer | RPMG | + LDS sample weighting |
| `latent_kitti_vio_tokens` | TokenizedPoseTransformer | CrossEntropy | Classification approach |
| `visual` | PoseTransformerVisual | RPMG | Vision-only ablation |
| `inertial` | PoseTransformerInertial | RPMG | IMU-only ablation |

---

## 12. File Reference

### Core Pipeline

| File | Purpose |
|------|---------|
| `cache_latent_features.py` | Stage 1: Encode raw data → cached .npy features |
| `train.py` | Stage 2: Hydra-based training entry point |
| `eval.py` | Evaluation entry point with checkpoints |

### Model Definitions

| File | Purpose |
|------|---------|
| `pose_transformer.py` | PoseTransformer, TokenizedPoseTransformer, Visual/Inertial variants |
| `vio_simple_dense_net.py` | VIOSimpleDenseNet baseline (sanity check) |

### Training Framework

| File | Purpose |
|------|---------|
| `vio_module.py` | `VIOLitModule` — standard PyTorch Lightning training module |
| `weighted_vio_module.py` | `WeightedVIOLitModule` — with per-sample importance weighting |
| `vio_datamodule.py` | `VIODataModule` — Lightning data module wrapper |
| `weighted_loss.py` | All loss function variants (MSE, MAE, RPMG, Tokenized, etc.) |

### Data Handling

| File | Purpose |
|------|---------|
| `KITTI_dataset.py` | KITTI dataset class with LDS weight computation |
| `latent_kitti_dataset.py` | Lightweight dataset loading cached .npy features |
| `custom_transform.py` | Image/IMU augmentation pipeline |

### Evaluation & Utils

| File | Purpose |
|------|---------|
| `kitti_eval.py` | Standard KITTI evaluation with trajectory plotting |
| `kitti_latent_eval.py` | Latent-based evaluation (encode → Transformer → evaluate) |
| `kitti_latent_tester.py` | Lightning-compatible latent tester wrapper |
| `kitti_metrics_calculator.py` | KITTI metric computation (t_rel, r_rel, RMSE) |
| `kitti_utils.py` | Pose math utilities (Euler↔Matrix, RMSE, trajectory distance) |
| `rpmg.py` | Riemannian Projected Manifold Gradient implementation |
| `tools.py` | SVD, orthogonalization, geodesic distance utilities |

### Infrastructure

| File | Purpose |
|------|---------|
| `pylogger.py` | Ranked logger for distributed training |
| `instantiators.py` | Hydra callback/logger instantiation helpers |
| `logging_utils.py` | Hyperparameter logging utilities |
| `vift_utils.py` | Task wrapper, extras, metric retrieval |

---

## 13. Usage Commands

### Stage 1: Cache Latent Features

```bash
# Cache training features
python cache_latent_features.py --split train --seq_len 11

# Cache validation features
python cache_latent_features.py --split val --seq_len 11
```

### Stage 2: Training

```bash
# Standard PoseTransformer training
python train.py experiment=latent_kitti_vio_tf

# Weighted training with RPMG loss
python train.py experiment=latent_kitti_vio_weighted_tf

# Override hyperparameters
python train.py experiment=latent_kitti_vio_tf trainer.max_epochs=300 data.batch_size=64

# Debug mode (quick sanity check)
python train.py debug=default
```

### Evaluation

```bash
# Evaluate a checkpoint
python eval.py ckpt_path="logs/train/runs/.../checkpoints/best.ckpt" model=latent_vio_tf

# Evaluate on specific sequences
python eval.py ckpt_path="best.ckpt" model.tester.val_seqs=['09','10']
```

---

## 14. Suggestions to Improve Accuracy

This section proposes concrete improvements that can boost accuracy **without increasing** (or even while **reducing**) the parameter count.

---

### 14.1. Architectural Improvements (Same or Fewer Parameters)

#### A. Replace Learnable Linear Projection with Grouped Linear Layers

**Current:** `Linear(768 → 128)` treats visual and inertial features identically.

**Improvement:** Use two separate smaller projections that respect the modality boundary:

```python
self.visual_proj = nn.Linear(512, 96)    # 512 × 96 = 49,152 params
self.inertial_proj = nn.Linear(256, 32)  # 256 × 32 = 8,192 params
# Total: 57,344 vs current 768 × 128 = 98,304 (41% fewer params)
# Output: 96 + 32 = 128 (same embedding dim)
```

**Why:** Each modality has different statistical properties. Separate projections let the model learn modality-specific transformations before fusion occurs in the attention layers.

#### B. Replace Sinusoidal Positional Embedding with Learnable Embeddings

**Current:** Fixed sinusoidal encoding.

**Improvement:** Use `nn.Embedding(max_seq_len, embedding_dim)` (only adds ~1,408 parameters for seq_len=11, dim=128).

**Why:** Learnable positional embeddings can capture task-specific temporal relationships that generic sinusoidal patterns cannot, and the parameter cost is negligible.

#### C. Add Relative Positional Bias to Attention

Instead of additive positional embeddings, inject a learnable relative position bias directly into the attention scores (similar to ALiBi or T5-style bias). This adds only `O(seq_len²)` = 121 parameters but dramatically improves distance-dependent attention patterns.

#### D. Use Pre-LayerNorm Instead of Post-LayerNorm

The default `nn.TransformerEncoderLayer` uses Post-LayerNorm. Switching to Pre-LayerNorm (applying LayerNorm before attention and FFN) is known to:
- Stabilize training with larger learning rates
- Converge faster and to lower loss values
- Require zero additional parameters

```python
nn.TransformerEncoderLayer(..., norm_first=True)
```

---

### 14.2. Loss Function Improvements (Zero Additional Parameters)

#### A. Multi-Task Weighted Loss with Learnable Balancing

Replace hand-tuned `angle_weight` with uncertainty-based automatic loss weighting (Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses"):

```python
log_sigma_rot = nn.Parameter(torch.zeros(1))   # +1 param
log_sigma_trans = nn.Parameter(torch.zeros(1))  # +1 param

loss = (1 / (2 * torch.exp(2 * log_sigma_rot))) * rotation_loss + log_sigma_rot
     + (1 / (2 * torch.exp(2 * log_sigma_trans))) * translation_loss + log_sigma_trans
```

This automatically learns the optimal rotation-vs-translation weighting during training.

#### B. Smooth L1 Loss (Huber Loss)

Replace MAE/MSE with `nn.SmoothL1Loss`:
- Behaves like L2 for small errors (stable gradients near zero)
- Behaves like L1 for large errors (robust to outliers)
- Zero additional parameters

#### C. Velocity Consistency Regularization

Add a penalty for physically implausible accelerations:

```python
velocity = poses[:, 1:, 3:] - poses[:, :-1, 3:]  # Finite differences
acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
consistency_loss = torch.mean(acceleration ** 2)
total_loss = pose_loss + λ * consistency_loss
```

This acts as a physics-based prior with zero additional parameters.

---

### 14.3. Training Strategy Improvements

#### A. Curriculum Learning

Start training with easier sequences (straight driving) and gradually introduce harder ones (sharp turns):

```python
# Epoch 1-50: Only train on samples with rotation < median
# Epoch 50-100: Train on all samples
# Epoch 100+: Over-sample hard samples (high rotation)
```

#### B. Exponential Moving Average (EMA) of Weights

Maintain an exponential moving average of model weights and use it for evaluation:

```python
ema_decay = 0.999
ema_weights = ema_decay * ema_weights + (1 - ema_decay) * current_weights
```

This typically improves validation performance by 5–15% with zero additional inference-time parameters.

#### C. Cosine Annealing with Warmup

Replace the current `CosineAnnealingWarmRestarts` with a linear warmup phase followed by cosine decay:

```
LR: 0.0 → 0.0001 (warmup, 10 epochs) → 0.000001 (cosine decay, remaining epochs)
```

Warmup prevents the randomly initialized attention weights from producing large, destructive gradients in early training.

#### D. Gradient Clipping

Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to prevent rare but catastrophic gradient explosions, especially with the RPMG loss.

---

### 14.4. Data-Level Improvements (Zero Additional Parameters)

#### A. Sliding Window with Overlap at Evaluation

**Current:** Non-overlapping windows during evaluation cause discontinuities at window boundaries.

**Improvement:** Use overlapping windows and average predictions in the overlap region:

```python
# Window stride = 1 instead of seq_len-1
# For each frame, average all predictions from windows that include it
```

#### B. IMU Noise Injection Augmentation

Add Gaussian noise to cached IMU features during training:

```python
noise = torch.randn_like(latent[:, :, 512:]) * 0.01  # Only on IMU portion
latent[:, :, 512:] += noise
```

This makes the model robust to real-world IMU sensor noise.

#### C. Temporal Flip Augmentation

Reverse the sequence order with negated translations:

```python
if random.random() < 0.5:
    latent = torch.flip(latent, dims=[0])   # Reverse sequence
    gt[:, 3:] = -gt[:, 3:]                   # Negate translations
```

This doubles the effective training data.

#### D. Sequence Length Randomization

During training, randomly crop each sequence to a shorter length (e.g., 7–11 frames). This:
- Acts as regularization
- Makes the model robust to varying input lengths
- Costs zero additional parameters

---

### 14.5. Inference Improvements (Zero Additional Parameters)

#### A. Test-Time Augmentation (TTA)

At inference, run the model on both the original and horizontally-flipped input, then average the predictions (flipping back the signs appropriately). This typically improves accuracy by 2–5%.

#### B. Multi-Scale Evaluation with History

Instead of a single window of 11 frames, maintain a rolling history buffer and let the model attend to longer contexts at test time (the causal mask and sinusoidal embeddings already support arbitrary lengths).

---

### Summary of Suggestions by Impact vs. Effort

| Suggestion | Impact | Effort | Params Change |
|------------|--------|--------|---------------|
| Pre-LayerNorm (`norm_first=True`) | ★★★ | ★ | 0 |
| Learnable positional embeddings | ★★★ | ★ | +1,408 |
| Velocity consistency regularization | ★★★ | ★★ | 0 |
| Smooth L1 loss | ★★ | ★ | 0 |
| Learnable loss weighting | ★★★ | ★ | +2 |
| EMA of weights | ★★★ | ★★ | 0 (at inference) |
| Grouped modality projections | ★★ | ★★ | -40,960 |
| Warmup + cosine decay | ★★ | ★ | 0 |
| Sliding window evaluation | ★★★ | ★★ | 0 |
| IMU noise augmentation | ★★ | ★ | 0 |
| Gradient clipping | ★★ | ★ | 0 |
| Curriculum learning | ★★ | ★★★ | 0 |

---

## License & References

- **RPMG:** Licensed under CC BY-NC 4.0.  Source: [github.com/JYChen18/RPMG](https://github.com/JYChen18/RPMG)
- **Label Distribution Smoothing:** Based on [Yang et al., "Delving into Deep Imbalanced Regression" (ICML 2021)](https://github.com/YyzHarry/imbalanced-regression)
- **Transformer Architecture:** Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
- **KITTI Benchmark:** Geiger et al., "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite" (CVPR 2012)
- **Hydra Config:** [github.com/facebookresearch/hydra](https://github.com/facebookresearch/hydra)
- **PyTorch Lightning:** [github.com/Lightning-AI/pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning)
