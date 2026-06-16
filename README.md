# Fertility Diagnosis ML Classification Task

Machine Learning project for binary classification of male fertility diagnosis using the Fertility Dataset.

## Overview

This project aims to build a machine learning model capable of predicting male fertility status from clinical and lifestyle-related features.

The problem is formulated as a **binary classification task**:

- **N (Normal)** → Normal fertility
- **O (Altered)** → Altered fertility

Due to the strong class imbalance present in the dataset, special attention was given to model evaluation and selection.

---

## Dataset

The project uses the **Fertility Dataset**, composed of:

- 100 samples
- 9 input features
- 1 binary target variable

### Features

| Feature | Description |
|----------|------------|
| Season | Season when the analysis was performed |
| Age | Patient age (normalized) |
| Childish diseases | History of childhood diseases |
| Accident or trauma | Previous accidents or trauma |
| Surgical intervention | Previous surgical interventions |
| High fevers | Fever episodes during the last 3 months |
| Alcohol consumption | Frequency of alcohol consumption |
| Smoking habit | Smoking behavior |
| Sitting hours | Daily sitting time (normalized) |

Target:

- **N** = Normal fertility
- **O** = Altered fertility

---

## Class Imbalance

The dataset is highly imbalanced:

- Normal (N): ~88%
- Altered (O): ~12%

Because of this, standard accuracy alone is not an appropriate metric.

The following strategies were adopted:

- Stratified train/test split
- Balanced Accuracy as the primary evaluation metric
- `class_weight='balanced'` where supported

---

## Data Preprocessing

### Train/Test Split

- 75% Training Set
- 25% Test Set
- `random_state=0`
- Stratified sampling

### Feature Scaling

Features are standardized using:

```python
StandardScaler()
```

The scaler is fitted only on the training set to avoid data leakage.

---

## Models Evaluated

Four machine learning models were compared using Grid Search with 5-Fold Cross Validation.

### 1. Logistic Regression

Hyperparameters:

- Penalty: L1, L2
- C ∈ {1e-5, 5e-5, 1e-4, 5e-4, 1}

Best CV Balanced Accuracy:

```text
0.564
```

---

### 2. Support Vector Machine (SVM)

Hyperparameters:

- Kernel: Linear, RBF
- C ∈ {0.1, 1, 10, 100}
- Gamma ∈ {scale, auto, 0.01, 0.1}

Best configuration:

```text
kernel = rbf
C = 1
gamma = auto
```

Best CV Balanced Accuracy:

```text
0.624
```

---

### 3. Random Forest

Hyperparameters:

- n_estimators ∈ {50, 100, 150}
- max_depth ∈ {3, 5, 7, None}
- min_samples_split ∈ {2, 5}

Best CV Balanced Accuracy:

```text
0.540
```

---

### 4. Multi-Layer Perceptron (MLP)

Hyperparameters:

- hidden_layer_sizes ∈ {n/2, n, 2n}
- alpha ∈ {0.0001, 0.001, 0.01}
- learning_rate_init ∈ {0.001, 0.01, 0.1}

Best CV Balanced Accuracy:

```text
0.597
```

---

## Model Selection

Comparison of cross-validation results:

| Model | Balanced Accuracy |
|---------|---------|
| SVM | 0.624 |
| MLP | 0.597 |
| Logistic Regression | 0.564 |
| Random Forest | 0.540 |

The **Support Vector Machine (RBF Kernel)** achieved the best validation performance and was selected as the final model.

---

## Final Test Results

Test set size:

```text
25 samples
```

Classification Report:

| Class | Precision | Recall | F1-score |
|---------|---------|---------|---------|
| Normal (N) | 0.91 | 0.95 | 0.93 |
| Altered (O) | 0.50 | 0.33 | 0.40 |

Overall metrics:

```text
Accuracy           = 0.88
Balanced Accuracy  = 0.644
```

Confusion Matrix:

```text
[[21 1]
 [ 2 1]]
```

---

## Technologies

- Python
- NumPy
- Pandas
- Scikit-Learn

Main Scikit-Learn components:

- StandardScaler
- GridSearchCV
- LogisticRegression
- SVC
- RandomForestClassifier
- MLPClassifier

---

## Project Structure

```text
.
├── fertility_Diagnosis.txt
├── train_fertility.py
├── report.pdf
└── README.md
```

---

## How to Run

Install dependencies:

```bash
pip install numpy pandas scikit-learn
```

Run the training script:

```bash
python train_fertility.py
```

The script will:

1. Load the dataset
2. Split train/test data
3. Scale features
4. Perform Grid Search on all models
5. Select the best model
6. Train the final classifier
7. Evaluate on the test set

---

## Conclusions

Despite the small dataset size and severe class imbalance, the project demonstrates that machine learning techniques can identify fertility-related patterns better than random guessing.

Among the tested approaches, an **RBF Support Vector Machine** provided the best balance between sensitivity and overall performance, achieving a **Balanced Accuracy of 0.644** on the test set.

Future improvements could include:

- Larger datasets
- Additional medical features
- Advanced resampling techniques (SMOTE)
- Ensemble methods
- Nested cross-validation

---

## Author

**Hartaj Bajwa Singh**

Bachelor's Degree in Computer Engineering  
University of Modena and Reggio Emilia (UNIMORE)
