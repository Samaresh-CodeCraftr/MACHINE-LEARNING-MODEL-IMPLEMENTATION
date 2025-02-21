# MACHINE-LEARNING-MODEL-IMPLEMENTATION
*COMPANY*: CODTECH IT SOLUTIONS
*NAME*: SAMARESH DAS
*INTERN ID*: CT08PCM
*DOMAIN*: PYTHON PROGRAMMING
*DURATION*: 4 WEEKS
*MENTOR*: NEELA SANTHOSH KUMAR
# SMS Spam Detection Using Naïve Bayes

## Description
This Python project implements an **SMS Spam Detection Model** using **Naïve Bayes** classification. The script processes a dataset containing SMS messages labeled as `spam` or `ham` (not spam), applies **TF-IDF vectorization**, and trains a **Multinomial Naïve Bayes model** to classify messages.

## Features
✅ Reads and processes an SMS dataset (`spam.csv`).
✅ Uses **TF-IDF Vectorization** to convert text into numerical features.
✅ Splits data into training and testing sets.
✅ Trains a **Multinomial Naïve Bayes** model.
✅ Evaluates model performance using **accuracy score, classification report, and confusion matrix**.
✅ Generates a **heatmap visualization** of the confusion matrix.

## Requirements
- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Script
```bash
python spam_detection.py
```

## Input Dataset
- The script uses a **CSV dataset (`spam.csv`)**, which contains SMS messages labeled as `spam` or `ham`.

## Output
- **Accuracy score** of the model.
- **Classification report** (precision, recall, F1-score).
- **Confusion matrix** visualization using `seaborn`.

## Example Output
```
Accuracy: 0.98
Classification Report:

              precision  recall  f1-score  support

           0       0.97    1.00    0.98      965
           1       1.00    0.79    0.88      150

    accuracy       0.97                    1115
   macro avg       0.98    0.90    0.93      1115
weighted avg       0.97    0.97    0.97      1115

Confusion Matrix:
[[965   0]
 [ 31 119]]
```


