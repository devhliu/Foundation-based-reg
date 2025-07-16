# 🧠 Are Vision Foundation Models Ready for Out-of-the-Box Medical Image Registration?

This is the **official repository** for our paper:

**_"Are Vision Foundation Models Ready for Out-of-the-Box Medical Image Registration?"_** [📄 Paper Link]

**Author**: Hanxue Gu*, Yaqian Chen*, Nick Konz, Qihang Li and Maciej A. Mazurowski

---

## 🔍 Overview

This repository implements a **training-free (zero-shot)** medical image registration pipeline using vision foundation models as feature encoders. We evaluate five different models:

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [MedSAM](https://github.com/bowang-lab/MedSAM)
- [SSL-SAM](https://github.com/mazurowski-lab/finetune-SAM/)
- [MedCLIP-SAMv2](https://github.com/healthx-lab/medclip-samv2)

Each model is used to extract image features that are then aligned using a training-free registration optimization pipeline—**no fine-tuning required**. Though our paper is heavily focused on breast image registration, we excited to see how it can be extended into other tasks!

---

## ⚙️ Step-by-Step Tutorial

### ✅ Step 1: Environment Setup

```bash
# Create and activate a clean conda environment
conda create -n dinov2 python=3.10 -y
conda activate dinov2

# Install PyTorch with CUDA 11.7
pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install dependencies
pip install torchmetrics==0.10.3 timm opencv-python

# Install DINOv2
pip install git+https://github.com/facebookresearch/dinov2.git

# Install other required libraries
pip install -r requirements.txt
```


---

### 📁 Step 2: Dataset Preparation

- Organize your image volumes as `.nii.gz` files under a directory, for example: `sample_dataset_dir/`
- Create a `task.csv` file that defines each image pair for registration, with columns:
  - `mov_volume`: moving image
  - `fix_volume`: fixed image

✅ You can refer to the provided `sample_dataset_dir/` for a template and example setup.
- You can download our prepared examples with two pre-contrast breast MRIs from the same patient taken at different dates from [Google drive](https://drive.google.com/drive/folders/16m2xlq4N4p5EE5va8LKBI4HohbG9vWzv?usp=drive_link).

---

### Step 3: Encoders preparation
- For all models, you can find the model weights to be downloaded in their original repo. 
- For SAM, please choose to download the Vit-b version.
- For SSLSAM, you can find the model weights under [SSLSAM](https://drive.google.com/drive/folders/1JAoy-Mh5QgxXsjWtQhMjOX16dN1kytLQ).


### ⚙️ Step 3: Configure Your Experiment

Edit the configuration file for your experiment. You can choose different models and adjust paths:

Key config fields:
- `exp_note`: A name for your experiment
- `model_ver`: One of:
  - `'sam'`
  - `'dino-v2'`
  - `'sslsam'`
  - `'medsam'`
  - `'biomedclip-sam'`
  - `'MIND'` (classic handcrafted feature)
- `data_dir`: The path to your dataset
- `save_feature`: Set to `'True'` to save extracted features for reuse

📌 Example config files are provided in the repo to help you get started quickly.

---

### 😃 Step 4: Run Registration Code
```python
python inference_reg.py --cfg config-dinov2-task1.py

```


### 📊 Step 5: Validation and Visualization

You can validate and visualize registration performance using the following tools:

- 📈 **Visualize registration result**:
  - Open and run `vis_result.ipynb`
- 🩻 **Overlay masks on registered volumes**:
  - See **Part 1** in `eval_dsc.ipynb`
- 📉 **Calculate Dice Similarity Coefficient (DSC)** across registered cases:
  - See **Part 2** in `eval_dsc.ipynb`


---

## 🙏 Acknowledgments

This work is heavily developed based on the excellent [DINO-reg](https://github.com/RPIDIAL/DINO-Reg) repository.
Big thanks to the original authors for their contributions to cross-modality medical image registration.

---

## 🧩 To-Do List

We are currently releasing the **zero-shot, image-only** registration pipeline.

### ✅ Current:
- Zero-shot registration with image pairs only

### 🔜 Coming Soon:
- [ ] Mask-guided registration support

---

Feel free to ⭐️ this repo, and cite our work if you find this repo helpful!

```


```