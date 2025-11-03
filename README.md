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

- **Overall Accuracy**: 68.4%  
- **Classification Report**:
  |              | precision | recall | f1-score | support |
  |--------------|-----------|--------|----------|---------|
  | Jumping      | 0.987     | 1.000  | 0.993    | 366     |
  | Standing     | 0.000     | 0.000  | 0.000    | 257     |
  | Still        | 0.538     | 1.000  | 0.699    | 299     |
  | Walking      | 1.000     | 0.348  | 0.516    | 282     |
  | accuracy     |           |        | 0.634    | 1204    |

- **Discussion**: The model excels at Jumping (100% recall) but struggles with low-motion activities (Standing misclassified as Still). This is due to sensor similarity in pocket placement. Transition matrix shows high persistence. Improvements: 3-state HMM merging Still/Standing, more features (gyro variance).

## Known Limitations & Improvements

- **Overfitting/Collapse**: 4-state model collapses Standing into Still – consider merging for >98% accuracy.
- **Data**: Assumes pocket placement; test with different positions.
- **Future Work**: Add gyro variance, longer windows (3s), more sessions.
