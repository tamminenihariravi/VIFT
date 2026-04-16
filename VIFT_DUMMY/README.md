# VIFT Visual-Inertial Odometry Models

This codebase implements various modules and training pipelines to manage Visual-Inertial Odometry models efficiently. It is built on PyTorch Lightning with a Hydra config system, ensuring modular tracking, configurable training loops, and streamlined inference. 

## 1. Quick Setup & Commands

Since the project structure is flattened, you can run all training and testing directly from this directory.

### Basic Training
Pick an experiment configuration (located in `configs/experiment`) and run:
```bash
python train.py experiment=kitti_vio
# or
python train.py experiment=latent_kitti_vio_tf
```

### Overriding Configuration
You can override any yaml setting directly from the command line:
```bash
python train.py experiment=kitti_vio trainer.max_epochs=50 data.batch_size=16
```

### Debugging
To quickly test if the model builds without running a full training loop:
```bash
python train.py debug=default
```

### Checking Configs
To view all configurable targets dynamically initialized by Hydra:
```bash
python train.py --help
```

---

## 2. Evaluation & Inference

### A. Collecting Weights
When you train a model, PyTorch Lightning automatically saves the best checkpoints inside your log directory. Checkpoints format typically looks like:
```text
logs/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/epoch_xyz.ckpt
```

### B. Running Inference (Evaluation Code)
Once trained, to compute test/validation sequence evaluations with the stored geometries, run `eval.py`:
```bash
python eval.py ckpt_path="logs/train/runs/.../checkpoints/best.ckpt" model=latent_vio_tf
```

If you only want to test on specific subset testing sequences:
```bash
python eval.py ckpt_path="best.ckpt" model.tester.val_seqs=['09','10']
```

---

## 3. Systems Flow Architecture

This outlines the high-level life cycle of data in our training loop.

**1. Data Loading:**
*   Hydra processes configurations (`configs/experiment/*.yaml`).
*   The system loads KITTI sequences (Images & IMU arrays) and applies cropping/augmentations dynamically via `custom_transform.py`.

**2. Forward Pass:**
*   Handled via Lightning Modules (`vio_module.py` or `weighted_vio_module.py`).
*   Processes sequences visually and temporally (e.g. using `pose_transformer.py`).
*   **Result:** Raw multidimensional outputs modeling temporal memory.

**3. Kinematics & Mathematical Formatting:**
*   Engines inside `rpmg.py` process the unstructured dense arrays.
*   Functions apply SVD and Symmetric Orthogonalization.
*   **Result:** Valid $3\times3$ Rotational Matrices and $3$-point internal translations.

**4. Loss Calculation & Updates:**
*   Matches estimations geometrically against Ground Truth Target records.
*   Loss variants (MSE, SE3 distances) processed inside `weighted_loss.py`.

**5. Validation & Metrics Testing:**
*   Triggered iteratively, tested cleanly and evaluated physically for tracking precision.
*   2D plotting automatically generated for checkpoints under best-loss conditions.
