<p align="center">
  <h1 align="center">Thermal Image Super-Resolution via Transfer Learning on Multi-Domain Datasets</h1>
</p>

<p align="center">
  Adapted from SGNet for Multispectral Satellite and Aerial Imagery
</p>

---

## 3D CAD of our product
For further details of our full design, please click on the following link: https://cad.onshape.com/documents/c03dc00e69275b6d8de646d9/w/dff503eb9fe2f4284a8bd7b3/e/ad6678004eb5e5bc5f7c590d

![Our product](3d_cad.png)

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
- **KAIST Multispectral**
- **VEDAI**
- **SugarBeet2016**

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
python train_kaist_from_flir.py \
  --pretrained [FLIR_MODEL_PATH] \
  --scale 4 \
  --num_feats 32 \
  --epoch 100 \
  --lr 0.00005 \
  --batchsize 4 \
  --max_samples 1000
```

## Stage 3: Fine-tuning on VEDAI

```bash
python train_vedai_from_kaist.py \
  --pretrained [KAIST_MODEL_PATH] \
  --scale 4 \
  --num_feats 32 \
  --epoch 100 \
  --lr 0.00005 \
  --batchsize 4 \
  --max_samples 1000
```

## Stage 4: Fine-tuning on Sugar beet 2016
```bash
python train_sugarbeet_from_vedai.py \
  --pretrained [VEDAI_MODEL_PATH] \
  --scale 4 \
  --num_feats 32 \
  --epoch 100 \
  --lr 0.00005 \
  --batchsize 4 \
  --max_samples 1000
```


## Citation

If you find this work useful, please consider citing the original SGNet paper: https://arxiv.org/pdf/2312.05799
  
