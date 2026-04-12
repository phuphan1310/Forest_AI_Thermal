<p align="center">
  <h1 align="center">Thermal Image Super-Resolution via Transfer Learning on Multi-Domain Datasets</h1>
</p>

<p align="center">
  Adapted from SGNet for Multispectral Satellite and Aerial Imagery
</p>

---

## 📌 Overview

This repository contains our implementation of **thermal image super-resolution** based on the **SGNet architecture**, enhanced through **sequential transfer learning across multiple datasets** for improved generalization in remote sensing applications.

---

## ⚙️ Dependencies

Install the following dependencies:

```bash
Python==3.11.5
PyTorch==2.1.0
numpy==1.23.5
torchvision==0.16.0
opencv-python==4.8.1
scipy==1.11.3
tqdm==4.65.0
Pillow==10.0.1
matplotlib==3.8.0
```

# Datasets

The model is sequentially trained and fine-tuned across the following datasets:

- **FLIR ADAS**
- **FLAME 3**

---

# Models

All pretrained models can be accessed upon request.  

Our model undergoes **sequential fine-tuning across four datasets**, with progressive improvement in performance at each stage.

---

# Training Strategy

We employ a multi-stage transfer learning pipeline to progressively improve generalization.

---

## Stage 1: Initial Training on FLIR

```bash
python train_flir.py \
  --scale 4 \
  --num_feats 32 \
  --epoch 200 \
  --batchsize 4 \
  --max_samples 1000
  ```

## Stage 2: Fine-tuning on Kaist

```bash
python train_flam3_from_flir.py \
  --pretrained [FLIR_MODEL_PATH] \
  --scale 4 \
  --num_feats 32 \
  --epoch 100 \
  --lr 0.00005 \
  --batchsize 4 \
  --max_samples 1000
```


## Citation

If you find this work useful, please consider citing the original SGNet paper: https://arxiv.org/pdf/2312.05799
  
