<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=🖼️%20CNN%20CIFAR-10%20Classifier&fontSize=44&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Real-time%20Image%20Classification%20·%20FastAPI%20+%20PyTorch%20·%20HuggingFace%20Spaces&descAlignY=60&descAlign=50" width="100%"/>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/ronakrajput8882/cifar10-classifier)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**🚀 [Live Demo — Try it Now!](https://ronakrajput8882-cifar10-classifier.hf.space/)**

</div>

---

## 📌 Project Overview

A **custom CNN image classifier** trained from scratch on CIFAR-10, served via a **FastAPI backend** with a fully custom HTML/JS frontend — no Gradio, no Streamlit. Deployed as a Docker container on **HuggingFace Spaces**.

Upload any image → get an instant prediction with **top-3 class probabilities** and confidence scores across all 10 classes.

> 🎯 **~75.1% test accuracy** on CIFAR-10 using a custom 3-layer CNN built entirely in PyTorch.

---

## 📂 Dataset

| Property | Details |
|:---|:---|
| **Dataset** | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) |
| **Total Images** | 60,000 (50K train / 10K test) |
| **Image Size** | 32 × 32 × 3 (RGB) |
| **Classes** | 10 balanced classes |
| **Source** | `torchvision.datasets.CIFAR10` |

**Classes:** `airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

---

## 🔄 Pipeline Workflow

```
Image Upload → Resize (32×32) → Normalize → CNN Forward Pass → Softmax → Top-3 Predictions → JSON Response
```

### 1️⃣ Preprocessing
- Input image resized to **32×32** using `transforms.Resize`
- Normalized with mean `(0.5, 0.5, 0.5)` and std `(0.5, 0.5, 0.5)` → pixel values mapped to `[-1, 1]`
- Converted to PyTorch tensor and batched with `unsqueeze(0)`

### 2️⃣ Model Inference
- Single forward pass through the CNN
- `torch.softmax` applied to logits → probability distribution
- `torch.topk(probs, 3)` extracts top-3 predictions

### 3️⃣ API Response
- FastAPI `/predict` endpoint returns: top class, emoji, confidence %, top-3 predictions, and full probability distribution for all 10 classes

---

## 🤖 Model Architecture ⭐ Best Model

### Custom CNN — 3 Conv Layers

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),   # 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),  # 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), # 8×8 → 4×4
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
```

- **Feature maps:** 32 → 64 → 128 filters (progressive depth)
- **Spatial reduction:** 32×32 → 16×16 → 8×8 → 4×4 via MaxPooling
- **FC layers:** 2048 → 256 → 10 (output logits)
- **Inference:** CPU-compatible, single forward pass, no TTA

---

## 📊 Results

| Metric | Value |
|:---|:---:|
| **Test Accuracy** | **~75.1%** |
| **Architecture** | Custom CNN (3 Conv + 2 FC) |
| **Parameters** | ~2.1M |
| **Input Size** | 32 × 32 × 3 |
| **Output** | 10-class softmax |
| **Inference Mode** | CPU (no GPU required) |

---

## 🔍 Key Insights

- 🧠 **Progressive filter doubling** (32 → 64 → 128) consistently improves feature extraction on CIFAR-10 without overfitting at this scale
- 📉 **Resolution bottleneck** is the primary accuracy ceiling — CIFAR-10's 32×32 images lose fine-grained detail, making classes like `cat` vs `dog` genuinely hard even for CNNs
- ⚠️ **Softmax overconfidence** is real — the model outputs high confidence even on out-of-distribution images; temperature scaling would help
- 🚀 A ResNet-18 backbone on the same dataset would push accuracy to **~90–93%**, confirming the custom CNN is strong for its parameter count
- 🐸 `frog`, `ship`, and `airplane` are typically the easiest classes due to distinct color distributions; `cat` and `dog` are the hardest

---

## 🗂️ Repository Structure

```
cifar10-classifier/
│
├── app.py                  # FastAPI backend — model loading + /predict endpoint
├── index.html              # Custom frontend UI (drag & drop + results display)
├── cnn_cifar10.pth         # Trained model weights
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container config for HF Spaces
├── limitations.txt         # Known model limitations & future improvements
└── README.md               # This file
```

---

## 🚀 Quick Start

### Run Locally

```bash
# Clone the repo
git clone https://github.com/ronakrajput8882/CNN-Image-Classifier.git
cd CNN-Image-Classifier

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
# → Open http://localhost:7860
```

### Run with Docker

```bash
docker build -t cifar10-classifier .
docker run -p 7860:7860 cifar10-classifier
```

### Use the Live Demo

```
🌐 https://ronakrajput8882-cifar10-classifier.hf.space/
```

---

## 🧠 Key Learnings

- Serving a PyTorch model with **FastAPI** is more flexible and production-ready than Gradio/Streamlit for custom UIs
- **Docker on HuggingFace Spaces** gives full control over the runtime environment — no SDK lock-in
- CIFAR-10's 32×32 resolution is a hard accuracy ceiling for custom CNNs; modern architectures use **data augmentation** (RandomCrop, HorizontalFlip, Cutout) to push past 90%
- **Softmax probabilities are not calibrated** — a 95% confidence score ≠ 95% correct; always mention this to end users
- Building the frontend from scratch (vs Gradio) teaches you exactly what the model API contract looks like in production

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| **PyTorch** | Model definition, training, inference |
| **torchvision** | CIFAR-10 dataset, image transforms |
| **FastAPI** | REST API backend (`/predict` endpoint) |
| **uvicorn** | ASGI server |
| **Pillow** | Image loading and RGB conversion |
| **Docker** | Containerization for HF Spaces deployment |
| **HTML/CSS/JS** | Custom frontend UI |

---

<div align="center">

### 🌐 Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronakrajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>
