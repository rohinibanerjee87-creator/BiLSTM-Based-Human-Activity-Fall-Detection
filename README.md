# Deep Learning Based Human Activity Recognition and Fall Detection Using BiLSTM

A deep learning project that detects falls and recognizes daily activities from wearable sensor data (accelerometer and gyroscope) using a Bidirectional LSTM (BiLSTM) network.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohinibanerjee87-creator/BiLSTM-Based-Human-Activity-Fall-Detection/blob/main/Deep_Learning_Based_Human_Activity_Recognition_and_Fall_Detection_Using_BiLSTM.ipynb)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)

## Overview
Falls are a major health risk, especially for elderly people. This project uses motion sensor signals to automatically tell the difference between normal daily activities (ADL) and falls.

## Datasets
| Dataset | Sensors | Task |
|---|---|---|
| Preprocessed sensor data | Accelerometer + Gyroscope | ADL vs Fall |
| UP-Fall (wrist) | Wrist accelerometer | ADL vs Fall |
| UniMiB (all 8 classes) | Accelerometer | 7-class activity recognition |

## Methodology
1. Data collection and preprocessing
2. Windowing (window size = 50)
3. BiLSTM model training
4. Evaluation using accuracy, precision, recall, F1-score and confusion matrix

## Results
Evaluated on a held-out test set, split by subject so the same person never appears in both training and testing.
| Task | Accuracy | Macro F1 | Fall Recall | Fall F1 |
|---|---|---|---|---|
| ADL vs Fall (sensor data, ACC + GYR) | 99% | 0.99 | 1.00 | 0.98 |
| ADL vs Fall (UP-Fall, wrist accelerometer) | 96% | 0.73 | 0.71 | 0.47 |
| Activity recognition (UniMiB, 7 classes) | 99% | 0.97 | – | – |

## How to Run
1. Click the **Open in Colab** badge above.
2. Upload the dataset CSV files.
3. Click **Runtime → Run all**.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, PyTorch, BiLSTM, Matplotlib

## Authors
- Rohini Banerjee
- Imon Panja, Tanusree Ghosh, Shalu Mandal, Srimoyee Samanta
