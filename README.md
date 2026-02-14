# Obesity Level Estimation — ML Assignment 2

## Problem Statement

The goal of this project is to build and compare **6 machine learning classification models** to estimate the **obesity level** of individuals based on their eating habits and physical condition.

This is a **multiclass classification** problem with **7 target classes** ranging from "Insufficient Weight" to "Obesity Type III". The project includes model training, evaluation using multiple metrics, and an interactive **Streamlit web application** for real-time predictions.

---

## Dataset Description

**Dataset:** [Estimation of Obesity Levels Based On Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) — UCI Machine Learning Repository

| Property | Details |
|----------|---------|
| **Source** | UCI Machine Learning Repository |
| **Instances** | 2,111 |
| **Features** | 16 input features + 1 target |
| **Task** | Multiclass Classification (7 Classes) |
| **Target Variable** | `NObeyesdad` |

### Feature Details

| Feature | Description |
|---------|-------------|
| **Gender** | Gender of the individual |
| **Age** | Age in years |
| **Height** | Height in meters |
| **Weight** | Weight in kg |
| **family_history_with_overweight** | Has a family member who is or was overweight? |
| **FAVC** | Frequent consumption of high caloric food |
| **FCVC** | Frequency of consumption of vegetables |
| **NCP** | Number of main meals |
| **CAEC** | Consumption of food between meals |
| **SMOKE** | Do you smoke? |
| **CH2O** | Consumption of water daily |
| **SCC** | Calories consumption monitoring |
| **FAF** | Physical activity frequency |
| **TUE** | Time using technology devices |
| **CALC** | Consumption of alcohol |
| **MTRANS** | Transportation used |

**Target Classes:**
1. Insufficient_Weight
2. Normal_Weight
3. Overweight_Level_I
4. Overweight_Level_II
5. Obesity_Type_I
6. Obesity_Type_II
7. Obesity_Type_III

---

## Models Used

Six classification models were trained and evaluated:

1. **Logistic Regression** (Multinomial)
2. **Decision Tree Classifier**
3. **K-Nearest Neighbor (KNN)**
4. **Naive Bayes (Gaussian)**
5. **Random Forest (Ensemble)**
6. **XGBoost (Ensemble)**

### Comparison Table

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.8818 | 0.9834 | 0.8803 | 0.8818 | 0.8808 | 0.8620 |
| **Decision Tree** | 0.9102 | 0.9704 | 0.9157 | 0.9102 | 0.9108 | 0.8958 |
| **KNN** | 0.8487 | 0.9731 | 0.8461 | 0.8487 | 0.8390 | 0.8256 |
| **Naive Bayes** | 0.5154 | 0.8390 | 0.5472 | 0.5154 | 0.4717 | 0.4556 |
| **Random Forest** | 0.9409 | 0.9949 | 0.9464 | 0.9409 | 0.9422 | 0.9315 |
| **XGBoost** | 0.9551 | 0.9978 | 0.9567 | 0.9551 | 0.9554 | 0.9477 |

### Observations

| ML Model Name | Observation about model performance |
|:---|:---|
| **Logistic Regression** | Serves as a strong baseline with ~88% accuracy. It effectively separates distinct classes but struggles with complex non-linear boundaries compared to tree-based methods. |
| **Decision Tree** | Captures non-linear relationships well (~91% accuracy) but has slightly lower AUC than ensemble methods, likely due to some overfitting on the training data. |
| **KNN** | Moderate performance (~85%). It is computationally expensive during inference and sensitive to the "curse of dimensionality" with 16 features. |
| **Naive Bayes** | Worst performer (~51%). The assumption of feature independence is clearly violated in this dataset (e.g., Weight vs BMI characteristics), leading to poor predictions. |
| **Random Forest** | Excellent performance (~94%). The ensemble of trees reduces variance and improves generalization, making it robust against noise. |
| **XGBoost** | **Best Performer (~95.5%)**. Gradient boosting sequentially corrects errors, achieving the highest Accuracy and AUC. It is the most reliable model for this classification task. |

---

## Project Structure

```
MLAssignment2/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── model/
│   ├── train_models.py             # Model training script
│   ├── *.pkl                       # Trained models and scalers
│   └── metrics_results.json        # Evaluation metrics
└── data/
    ├── obesity_data.csv            # Dataset
    └── test_data.csv               # Pre-split test set for demo
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (Optional — pre-trained models included)
This script processes the data, applies SMOTE for balancing, trains all 6 models, and saves the artifacts.
```bash
python model/train_models.py
```

### 3. Run the Streamlit App
Launch the interactive dashboard to view results and make predictions.
```bash
streamlit run app.py
```

### 4. Use the App
- **Model Evaluation**: View detailed metrics (Accuracy, Precision, Recall, F1, AUC, MCC) and Confusion Matrices.
- **Comparison Table**: Compare performance across all 6 models.
- **Predictions**: Upload a CSV file (format matching `test_data.csv`) to generate predictions for new data.

---

## Technologies Used

- **Python 3.11.4**
- **scikit-learn** — ML models and evaluation
- **XGBoost** — Gradient boosting classifier
- **Streamlit** — Interactive web application
- **pandas & numpy** — Data manipulation
- **matplotlib & seaborn** — Visualizations
