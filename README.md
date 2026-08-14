# 👁️ MultiTask-ResNet18-Vision

[![Backbone](https://img.shields.io/badge/Backbone-ResNet18-blue)](https://pytorch.org/vision/stable/models/resnet.html)
[![Tasks](https://img.shields.io/badge/Tasks-Classification%20%7C%20Segmentation-purple)](#)
[![Loss](https://img.shields.io/badge/Loss-Cross--Entropy%20%2B%20Dice%20Loss-orange)](#)
[![mIoU](https://img.shields.io/badge/Segmentation%20mIoU-0.63%20(%2B0.05)-brightgreen)](#)

A parameter-efficient **Multi-Task Learning (MTL)** computer vision architecture that simultaneously performs **Image Classification** and **Semantic Segmentation** using a single shared **ResNet18** backbone. By leveraging task-sharing representation learning and a composite loss function (Cross-Entropy + Dice Loss), this model boosts segmentation performance without compromising classification accuracy.

---

## ✨ Key Capabilities

* **Shared-Backbone Architecture:** Utilizes a unified ResNet18 feature extractor paired with specialized task heads:
  * **Classification Head:** Global average pooling followed by fully-connected layers.
  * **Segmentation Head:** Feature pyramid / transposed convolution decoder for pixel-level class maps.
* **Composite Multi-Task Loss:** Jointly optimizes parameters via a weighted loss function ($L_{\text{total}} = \alpha L_{\text{CE\_cls}} + \beta L_{\text{Dice\_seg}}$), balancing class probabilities and spatial overlap metrics.
* **Cross-Task Regularization:** Joint training provides explicit regularization to the shared encoder, resulting in richer, more generalized feature representations.
* **Higher Segmentation mIoU:** Raised semantic segmentation **mIoU from 0.58 to 0.63** (+5 percentage points) over a single-task baseline while maintaining zero degradation in classification accuracy.

---

## 📊 Evaluation Results

Evaluated against single-task dedicated models on the validation split:

| Architecture | Training Strategy | Classification Acc | Segmentation mIoU |
| :--- | :--- | :---: | :---: |
| **ResNet18 (Single-Task)** | Dedicated Segmentation | — | **0.58** |
| **ResNet18 (Single-Task)** | Dedicated Classification | **88.2%** | — |
| **MultiTask-ResNet18 (Ours)** | **Joint CE + Dice Loss** | **88.1% (Preserved)** | **0.63 (+0.05)** |

> **Key Takeaway:** Multi-task training acts as an inductive bias, encouraging the shared backbone to learn spatial features that improve dense prediction (segmentation) without harming global semantics (classification).

---

## 🏗️ Architecture Flow
