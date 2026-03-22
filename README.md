# 🧠 Diabetic Retinopathy Detection – Deep Learning Pipeline

## 📌 Overview

This project focuses on **automatic classification of diabetic retinopathy (DR)** severity levels from retinal fundus images using deep learning.

The task is based on the **APTOS 2019 Blindness Detection dataset**, where each image is classified into one of **5 ordinal classes**:

* 0 → No DR
* 1 → Mild
* 2 → Moderate
* 3 → Severe
* 4 → Proliferative DR

The main goal is to **maximize Quadratic Weighted Kappa (QWK)**, a metric suitable for ordinal classification.

---

## 🚀 Final Approach (High-Level Pipeline)

```
Image
↓
Preprocessing + Augmentation
↓
ResNet50 + EfficientNet-B4
↓
Test Time Augmentation (TTA)
↓
Score Averaging (Ensemble)
↓
Threshold Optimization
↓
Final Prediction
```

---

## ⚙️ Key Components

### 1. Data Preprocessing

* Images resized (224 / 320 / 380)
* Normalization using ImageNet stats
* Class imbalance handled using **class weights**

---

### 2. Data Augmentation (Critical)

To reduce overfitting and improve generalization:

* Horizontal & Vertical flips
* Rotation + Shift + Scale
* Brightness / Contrast adjustments
* Color jitter (Hue/Saturation)
* Blur / Sharpen

✅ Result: Significant improvement in validation performance

---

### 3. Models Used

#### 🔹 ResNet50

* Pretrained on ImageNet
* Added Dropout (0.4) to reduce overfitting
* Fine-tuned last block (layer4)

#### 🔹 EfficientNet-B4

* Higher capacity model
* Fine-tuned last 3–4 blocks
* Lower learning rate for stability

---

## 📉 Problems Faced & Solutions

### ❌ Problem 1: Severe Overfitting

**Symptoms:**

* Very high train accuracy (~97%)
* Low validation performance

**Solution:**

* Strong augmentation
* Dropout layer
* Freezing backbone initially

---

### ❌ Problem 2: Class Imbalance

**Symptoms:**

* Model biased toward majority classes

**Solution:**

* Used `compute_class_weight`
* Applied weights in CrossEntropyLoss

---

### ❌ Problem 3: Weak Generalization

**Symptoms:**

* Gap between train and validation metrics

**Solution:**

* Fine-tuning deeper layers
* Learning rate scheduling (CosineAnnealing)
* Early stopping

---

### ❌ Problem 4: Ordinal Nature Ignored

**Issue:**

* Treating labels as categorical loses ordering information

**Solution:**

* Converted outputs → **continuous scores**
* Applied **threshold optimization** instead of argmax

---

### ❌ Problem 5: Prediction Instability

**Solution: Test Time Augmentation (TTA)**

* Original image
* Horizontal flip
* Vertical flip
* Averaged predictions

---

### ❌ Problem 6: Model Limitations

Single model plateaued around ~0.87 QWK

**Solution: Ensemble Learning**

* Combined ResNet + EfficientNet
* Averaged continuous scores (not classes)

---

## 📈 Performance Improvements

| Stage                    | QWK        |
| ------------------------ | ---------- |
| ResNet Baseline          | 0.856      |
| + Augmentation + Dropout | 0.869      |
| EfficientNet Fine-Tuning | 0.872      |
| + Threshold Optimization | 0.890      |
| + TTA                    | 0.892      |
| ✅ Ensemble (Final)       | **0.8989** |

---

## 🧪 Key Techniques Used

* Transfer Learning
* Fine-Tuning
* Strong Data Augmentation (Albumentations)
* Class Imbalance Handling
* Learning Rate Scheduling
* Early Stopping
* Threshold Optimization
* Test Time Augmentation (TTA)
* Model Ensembling

---

## 💡 Important Insights

* Increasing image resolution **does not always improve performance**
* EfficientNet needs **proper fine-tuning** to outperform ResNet
* Threshold optimization can boost performance **without retraining**
* Ensemble gives **consistent final gain**

---

## 🏁 Conclusion

This project demonstrates how **systematic experimentation** and combining multiple techniques can significantly improve performance in medical imaging tasks.

The biggest gains came from:

* Regularization (augmentation + dropout)
* Proper fine-tuning
* Threshold optimization
* Ensemble learning

---

## 📂 How to Run

1. Install dependencies

```
pip install torch torchvision albumentations
```

2. Prepare dataset (APTOS 2019)

3. Train models:

* Train ResNet50
* Train EfficientNet-B4

4. Apply:

* Threshold optimization
* TTA
* Ensemble

---

## 🔥 Final Result

> Achieved **~0.898 QWK**, close to top competitive solutions.

---

## 📎 Notes

* Designed for **Kaggle competition setting**
* Easily extendable to other medical imaging tasks
* Can be improved further with:

  * Advanced architectures (ConvNeXt, ViT)
  * Better ensembling strategies
  * Pseudo-labeling

---

⭐ If you found this useful, consider starring the repo!
