# Chronic-Kidney
Chronic-Kidney
# 🧠 Chronic Kidney Disease Detection using Hybrid Machine Learning Algorithms

A GUI-based desktop application using **Tkinter** and **Python (Scikit-learn)** to diagnose **Chronic Kidney Disease (CKD)** by leveraging a hybrid ensemble of machine learning algorithms including **Random Forest**, **AdaBoost**, **Bagging**, **SVM**, and **Decision Trees**, with **PCA** for feature selection.

---

## 🩺 Project Overview

This project aims to detect Chronic Kidney Disease at early stages by applying a **hybrid machine learning model** on medical data. It provides an intuitive **Tkinter-based GUI** where users can:

- Upload and preprocess CKD datasets
- Apply PCA-based dimensionality reduction
- Train multiple ML models and compare performance metrics
- Visualize accuracy, precision, recall, and F1-score
- Upload test data and predict CKD or NO-CKD outcomes

---

## 🚀 Features

- ✅ Upload and preprocess CKD dataset
- ✅ PCA for feature selection
- ✅ Hybrid voting classifier combining RF, SVM, AdaBoost, Bagging, and DT
- ✅ Comparison with Naive Bayes, SVM, and stacking models
- ✅ Confusion matrix and performance metrics graphs
- ✅ GUI to interact and visualize predictions
- ✅ Supports test file input and batch predictions

---

## 🖥️ GUI Snapshots

| Dataset Upload | PCA Feature Selection | Hybrid Model Accuracy |
|----------------|------------------------|------------------------|
| ![upload](https://i.imgur.com/5jzUN7Y.png) | ![pca](https://i.imgur.com/v8zEy7g.png) | ![accuracy](https://i.imgur.com/NKQqKON.png) |

> 📸 *Note: Replace the image links with your own GitHub-hosted or local `/assets` folder screenshots.*

---

## 📦 Technologies Used

- **Language**: Python 3.7
- **GUI**: Tkinter
- **Libraries**:  
  - `scikit-learn` (SVM, RandomForest, AdaBoost, Bagging, Voting, PCA)
  - `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
  - `Tkinter`, `LabelEncoder`
- **ML Techniques**:
  - Ensemble Voting Classifier
  - Stacking (Naive Bayes Tree)
  - PCA (Principal Component Analysis)

---

## ⚙️ How to Run

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/CKD-Hybrid-ML-Detection.git
   cd CKD-Hybrid-ML-Detection
