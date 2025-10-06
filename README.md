# ❤️ ECG Atrial Fibrillation (AFib) Detection

This project develops a **machine learning pipeline** to detect **Atrial Fibrillation (AFib)** from ECG signals using the **MIT-BIH Arrhythmia Database**.  
It showcases end-to-end work: preprocessing raw ECG data, feature extraction, model building, and evaluation.

---

## 🚀 Features
- **Signal Preprocessing**
  - Sliding window segmentation of ECG signals
  - Frequency-domain analysis using **Welch’s Power Spectral Density**
  - Extraction of statistical features (mean, variance, skewness, kurtosis)

- **Machine Learning Pipeline**
  - Logistic Regression baseline
  - Neural Network classifier (TensorFlow/Keras)
  - Standardization with **StandardScaler**

- **Evaluation Metrics**
  - Confusion Matrix
  - ROC Curve & AUC
  - Precision, Recall, F1-score
  - Best threshold optimization (F1-based)

---

## ⚙️ Tech Stack
- **Python 3.11**
- **NumPy / SciPy**
- **scikit-learn**
- **TensorFlow / Keras**
- **Matplotlib**

---

## 📊 Results
- Built a **baseline logistic regression** model with strong AUC
- Improved classification performance using a **neural network**
- Demonstrated interpretability of both simple and deep models

---

## 📂 Files
- `ecg_feature_extraction.py` → preprocessing & feature extraction
- `afib_logreg.ipynb` → Logistic Regression notebook
- `afib_neuralnet.ipynb` → Neural Network notebook
- `mitdb_features_v1.npz` → preprocessed dataset

---

## 🔮 Next Steps
- Add **CNN-based deep learning** for direct raw ECG waveform analysis
- Expand dataset with other arrhythmia types
- Deploy as a **real-time ECG classifier**

---

## 📚 Dataset
- MIT-BIH Arrhythmia Database ([PhysioNet](https://physionet.org/content/mitdb/1.0.0/))  

---

## 🧑‍💻 Author
Adi Wiesel  
[GitHub Profile](https://github.com/AdiWiesel)
