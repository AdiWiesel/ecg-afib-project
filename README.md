# ❤️ ECG Atrial Fibrillation (AFib) Detection

This project develops a **machine learning and deep learning pipeline** to detect **Atrial Fibrillation (AFib)** from ECG signals using the **MIT-BIH Arrhythmia Database**.  
It demonstrates the full journey from raw clinical signals → feature extraction → classical ML models → **1D Convolutional Neural Network (CNN)** for state-of-the-art detection.

---

## 🚀 Pipeline Overview
1. **Clinical Context**
   - Notebook on ECG basics (P wave, QRS, RR intervals, AFib interpretation).
   - Bridges medical knowledge with ML modeling.

2. **Data Preparation**
   - Extract ECG signals from **MIT-BIH Arrhythmia Database** (PhysioNet).
   - Apply **high-pass filtering** and **z-normalization**.
   - Segment into **5-second windows** with 2.5s stride.
   - Label windows as **AFib vs Normal** using annotation coverage.

3. **Feature Engineering**
   - **Time-domain features**: mean, std, RMS, skew, kurtosis.
   - **Frequency-domain features** via Welch PSD:
     - Bandpower (0.5–5 Hz, 5–15 Hz, 15–40 Hz)
     - Spectral centroid
     - Dominant frequency
     - High/low power ratio
   - Saved to `mitdb_features_v1.npz`.

4. **Modeling**
   - **Logistic Regression** (baseline on handcrafted features).
   - **Shallow MLP** (fully connected NN using features).
   - **1D CNN** (end-to-end learning on raw ECG signals).
   - Addressed class imbalance with **balanced TF datasets** and **focal loss**.

5. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
   - Visualizations:
     - ROC and PR curves
     - Training/validation curves
     - Confusion matrices
     - Probability histograms

---

## 📊 Results

| Model                     | Input                | ROC-AUC (Test) | Notes |
|----------------------------|---------------------|----------------|-------|
| Logistic Regression        | Engineered features | ~0.85          | Good baseline |
| Shallow MLP (FC NN)        | Engineered features | ~0.90          | Improved with nonlinearities |
| **1D CNN**                 | Raw ECG waveform    | **~0.95+**     | Best performance, automatic feature learning |

✅ The **1D CNN** consistently outperformed feature-based methods by learning directly from the ECG waveform.  

---

## ⚙️ Tech Stack
- **Python 3.11**
- Data: `wfdb`, `NumPy`, `SciPy`
- ML: `scikit-learn`
- DL: `TensorFlow / Keras`
- Visualization: `Matplotlib`

---

## 📂 Repository Structure
- `ecg_basics.ipynb` → ECG interpretation notes (clinical context)  
- `dataset.ipynb` → Raw data processing & window extraction  
- `fit.ipynb` → Train/Val/Test splits & feature extraction  
- `MLP_shallow.ipynb` → Shallow fully-connected NN  
- `1d_CNN.ipynb` → End-to-end deep learning with CNN  
- `models.ipynb` → Model comparisons (LogReg, MLP, CNN)  
- `mitdb_windows_5s_binary_afib.npz` → Saved ECG dataset  
- `mitdb_features_v1.npz` → Extracted features  

---

## 🔮 Next Steps
- Tune CNN hyperparameters (filters, kernel size, learning rate).  
- Explore **RNN/LSTM** for sequence modeling of longer signals.  
- Extend classification beyond AFib → multiple arrhythmias.  
- Deploy as a **real-time ECG classification system**.

---

## 📚 Dataset
- **MIT-BIH Arrhythmia Database** ([PhysioNet](https://physionet.org/content/mitdb/1.0.0/))  

---

## 🧑‍💻 Author
Adi Wiesel  
[GitHub Profile](https://github.com/AdiWiesel)
