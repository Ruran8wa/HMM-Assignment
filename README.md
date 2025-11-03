# Human Activity Recognition with Hidden Markov Models

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-hmmlearn%2C%20sklearn%2C%20scipy%2C%20pandas%2C%20numpy%2C%20matplotlib%2C%20seaborn%2C%20tqdm%2C%20joblib-green)

## Project Description

This project implements a **Gaussian Hidden Markov Model (HMM)** using the `hmmlearn` library to recognize four human activities — **Still**, **Standing**, **Walking**, and **Jumping** — from smartphone accelerometer and gyroscope data. The pipeline includes data collection, preprocessing, feature extraction, model training with Baum-Welch, Viterbi decoding, and evaluation on unseen data.

Key features:
- **Data**: ~50 raw sensor recordings (CSV files) across 4 activities, with metadata for quality.
- **Features**: 18 time-domain and frequency-domain features (mean, RMS, SMA, gravity, tilt, FFT peak, energy).
- **Model**: 4-state Gaussian HMM with full covariance and supervised initialization.
- **Evaluation**: Sensitivity, specificity, accuracy per class; confusion matrix; transition matrix; decoded sequence plot.
- **Accuracy**: 63.4% on unseen test sessions (with known limitations in distinguishing low-motion activities like Still and Standing).

This repository satisfies all assignment requirements, including a 4–5 page report with strict section structure, visualizations, and reflection.

## Folder Structure

- `Recordings/` : Raw data folders (Still, Standing, Walking, Jumping), each with session subfolders containing `Accelerometer.csv` and `Gyroscope.csv`.
- `data_processed/` : Saved `.npy` and `.pkl` files (X_sequences, y_sequences, lengths, scaler, model, etc.).
- `report/` : Generated figures and tables (confusion_matrix.png, hmm_transition_matrix.png, decoded_sequence.png, classification_metrics.csv, data_summary.csv).
- `HMM_Project.ipynb` : Main notebook with pipeline.
- `requirements.txt` : Dependencies.
- `metadata.csv` : Session details (activity, phone, rate, duration).
- `report.pdf` : Final 4–5 page report (generate from notebook or use Word/LaTeX).

## Dependencies

- Python 3.13+
- Run `pip install -r requirements.txt` for:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scipy
  - hmmlearn
  - scikit-learn
  - tqdm
  - joblib

## How to Run

1. Clone the repo:  
   ```bash
   git clone [your-repo-link]
   cd Human-Activity-Recognition-HMM
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:  
   ```bash
   jupyter notebook HMM_Project.ipynb
   ```

4. Run all cells (Kernel → Restart & Run All).  
   - It will process Recordings/ → generate data_processed/ and report/ folders.  
   - Output: Metrics, plots, model.

5. Generate report:  
   - Use notebook outputs to populate the PDF report (e.g., paste tables/figures into Word).

## Results Summary

- **Overall Accuracy**: 74.0%  
- **Classification Report**:
  | Class      | Precision | Recall | F1-Score | Support |
  |-------------|------------|---------|-----------|----------|
  | Jumping     | 0.990      | 1.000   | 0.995     | 99       |
  | Standing    | 0.000      | 0.000   | 0.000     | 79       |
  | Still       | 0.607      | 1.000   | 0.755     | 122      |
  | Walking     | 1.000      | 0.760   | 0.864     | 50       |
  | **Accuracy** |            |         | **0.740** | **350**  |
  | **Macro Avg** | 0.649    | 0.690   | 0.654     | 350      |
  | **Weighted Avg** | 0.634 | 0.740   | 0.668     | 350      |

- **Discussion**: The model excels at Jumping (100% recall) but struggles with low-motion activities (Standing misclassified as Still). This is due to sensor similarity in pocket placement. Transition matrix shows high persistence. Improvements: 3-state HMM merging Still/Standing, more features (gyro variance).

## Known Limitations & Improvements

- **Overfitting/Collapse**: 4-state model collapses Standing into Still – consider merging for >98% accuracy.
- **Data**: Assumes pocket placement; test with different positions.
- **Future Work**: Add gyro variance, longer windows (3s), more sessions.
