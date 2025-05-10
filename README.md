
# 🌿 HerbNet – Herbal Leaf Classification Using Deep Learning

HerbNet is a deep learning-based image classification project that identifies herbal plant species based on leaf images. This project was developed as part of my Master’s thesis and applies both custom CNN and transfer learning models to achieve high-accuracy classification across 53 herbal plant classes.

---

## 🚀 Project Highlights

- 📚 Classified 53 herbal plant species using leaf images.
- 🤖 Built and trained a **custom Convolutional Neural Network (CNN)**.
- 🔁 Applied **transfer learning** using VGG16, VGG19, ResNet50, and InceptionV3.
- 📈 Achieved **95% accuracy** with custom CNN and over **99%** with VGG16/ResNet.
- 🧪 Evaluated using **ROC-AUC, precision, recall, F1-score**, and confusion matrices.
- 📊 Used image preprocessing, data augmentation, and early stopping for optimal training.

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV
- Matplotlib, Seaborn
- Google Colab / Kaggle Notebooks

---

## 📂 Dataset

- Sourced from a manually curated collection of herbal leaf images.
- Total Images: ~7,000+
- Number of Classes: 53 (Each representing a distinct herbal plant)
- Augmented using random flips, rotations, zoom, and brightness shifts.
- Images resized to 224x224 for model compatibility.

---

## 🏗️ Model Architectures

### 1. 🔨 Custom CNN Architecture
- 3 Convolutional Blocks
- ReLU Activations + MaxPooling
- Dropout + Batch Normalization
- Fully Connected Dense Layers
- Softmax Output for Multiclass Classification

### 2. 🎓 Transfer Learning
- Pre-trained on ImageNet
- Top layers fine-tuned on our herbal dataset
- Models used: `VGG16`, `VGG19`, `ResNet50`, `InceptionV3`

---

## 📈 Evaluation Metrics

| Model        | Accuracy | F1-Score | AUC Score |
|--------------|----------|----------|-----------|
| Custom CNN   | 95.3%    | 0.94     | 0.96      |
| VGG16        | 99.2%    | 0.99     | 0.99      |
| ResNet50     | 99.1%    | 0.98     | 0.99      |

- Metrics were computed per class and macro-averaged.
- Confusion matrix and classification reports included in the notebook.

--



## 🧪 How to Run

> All training and evaluation was performed using Google Colab.

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/herbnet
cd herbnet

Contact
Raj Singh
📧 rs8260049@gmail.com
🔗 LinkedIn
