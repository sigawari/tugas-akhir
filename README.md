# 🇮🇩 Indonesian Sign Language (BISINDO) Translation to Text

### using Spatial ResNet-2D and MediaPipe Landmark Representation

## 🧠 Project Overview

This project investigates the **engineered utilization of a 2D Residual Network (ResNet-2D)** for translating **Indonesian Sign Language (BISINDO)** into text using **landmark-based spatial representations extracted with MediaPipe Holistic**.

Unlike most existing approaches that rely heavily on **temporal sequence models** such as LSTM, GRU, or Transformers, this research explores whether **purely spatial convolutional models** can effectively recognize **isolated BISINDO signs** when temporal information is encoded implicitly through structured landmark representations.

The system focuses on **Isolated Sign Language Recognition (ISLR)** and targets **basic BISINDO vocabulary** commonly used in daily communication.

---

## 🎯 Research Objectives

- Investigate the feasibility of **ResNet-2D as a spatial-only model** for BISINDO recognition.
- Design a **multichannel landmark representation** that embeds motion information without explicit temporal modeling.
- Analyze the impact of different **landmark combinations** (pose, hands, face) on recognition performance.
- Provide empirical evidence that **2D CNNs can learn temporal patterns implicitly** through structured spatial inputs.

---

## 🔬 Methodology Overview

### 1. Data Acquisition

- Videos of isolated BISINDO gestures are recorded under controlled conditions.
- Each sample represents **one complete sign gesture** with fixed start and end boundaries.

### 2. Landmark Extraction

- **MediaPipe Holistic** is used to extract:

  - Body pose landmarks
  - Hand landmarks
  - Facial landmarks

- Each landmark is represented as normalized 2D coordinates.

### 3. Spatial Multichannel Representation

- For each landmark:

  - Spatial position: `(x, y)`
  - Motion features: `(dx, dy)` computed between consecutive frames

- Landmarks are arranged into a **2D matrix (time × landmark index)**.
- The final input tensor has the shape:

  ```
  (Batch, Channels, Time, Landmarks)
  Channels = [x, y, dx, dy]
  ```

### 4. Model Architecture

- Backbone: **ResNet-18 / ResNet-34 (2D)**
- Input layer modified to accept **4-channel non-RGB input**
- No LSTM, GRU, or Transformer is used
- Temporal dynamics are captured **implicitly** through spatial structure

### 5. Evaluation

- Classification metrics:

  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 🧩 Key Contributions

- Demonstrates that **ResNet-2D can be applied beyond RGB images** for structured landmark data.
- Introduces a **lightweight spatial alternative** to temporal-heavy sign language models.
- Provides insight into how **implicit temporal encoding** can reduce model complexity.
- Extends landmark-based BISINDO research beyond static image classification.

---

## ⚠️ Scope and Limitations

- Focuses exclusively on **Isolated Sign Language Recognition (ISLR)**.
- Vocabulary size is limited to a small set of basic BISINDO signs.
- Does not handle continuous sentence-level translation.
- Landmark depth (`z`) is intentionally excluded due to instability in monocular estimation.

---

## 🚀 Future Work

- Extend the approach to **Continuous Sign Language Recognition (CSLR)**.
- Explore hybrid spatial–temporal architectures for comparison.
- Increase dataset diversity (signers, lighting, camera quality).
- Investigate attention mechanisms on spatial landmark representations.

---

## 📄 Academic Context

This repository contains documentation, experimental code, and research materials for the undergraduate thesis of:

**Sikah Nubuahtul Ilmi**
Bachelor of Informatics Engineering (S-1)
Faculty of Industrial Technology
Institut Teknologi Sumatera (ITERA)

---

## 🏷️ Thesis Titles

**Bahasa Indonesia**

> _Reka Cipta Pemanfaatan ResNet-2D dalam Penerjemahan Bahasa Isyarat Indonesia (BISINDO) Berbasis Landmark MediaPipe_

**English**

> _Engineered Utilization of ResNet-2D for Indonesian Sign Language Translation Based on MediaPipe Landmarks_
