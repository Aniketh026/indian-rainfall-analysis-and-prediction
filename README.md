# 🌧️ Indian Rainfall Analysis and Prediction

### A Machine Learning Project Using 10 Regression and Classification Models

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Pandas](https://img.shields.io/badge/Data%20Analysis-Pandas-150458?logo=pandas)
![Status](https://img.shields.io/badge/Project-Completed-success)

## 📌 Project Overview

**Indian Rainfall Analysis and Prediction** is a Data Science and Machine Learning project that focuses on analysing historical rainfall patterns across different meteorological subdivisions of India.

The project uses historical rainfall data from **1901 to 2017** to identify rainfall trends, understand monthly and annual rainfall distributions, and predict annual rainfall using machine learning algorithms.

A total of **10 machine learning models** are implemented, including 5 regression models and 5 classification models, to compare their performance in rainfall prediction.

## 🎯 Project Objectives

- Analyse historical rainfall patterns across Indian subdivisions.
- Perform Exploratory Data Analysis (EDA).
- Identify relationships between monthly and annual rainfall.
- Analyse yearly rainfall trends and seasonal variations.
- Predict annual rainfall using regression algorithms.
- Classify rainfall into Low, Medium and High categories.
- Compare the performance of 10 machine learning models.
- Visualize rainfall patterns and model evaluation results.

## 📂 Dataset Information

**Dataset Name:** Sub_Division_IMD_2017.csv

**Dataset Source:** India Meteorological Department (IMD)

**Time Period:** 1901–2017

**Dataset Size:** 4,188 records and 19 columns (before preprocessing).

The dataset contains historical rainfall information for 36 meteorological subdivisions of India.

### Important Features

| Feature | Description |
|---|---|
| SUBDIVISION | Meteorological subdivision |
| YEAR | Year of rainfall observation |
| JAN–DEC | Monthly rainfall measurements |
| ANNUAL | Total annual rainfall |
| JF | January–February rainfall |
| MAM | March–May rainfall |
| JJAS | June–September rainfall |
| OND | October–December rainfall |

## 🛠️ Technologies & Libraries Used

- **Python** – Main programming language.
- **Pandas** – Data manipulation and cleaning.
- **NumPy** – Numerical computations.
- **Matplotlib** – Data visualization.
- **Seaborn** – Statistical visualization.
- **Scikit-learn** – Machine learning and model evaluation.

## 🔍 Project Workflow

1. Dataset Collection
2. Data Loading and Inspection
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. Train-Test Splitting
7. Regression Model Implementation
8. Classification Model Implementation
9. Model Evaluation and Comparison
10. Data Visualization and Results

### Data Preprocessing

The following techniques are used:

- Missing value handling using SimpleImputer.
- Numerical feature scaling using StandardScaler.
- Categorical encoding using OneHotEncoder.
- Feature engineering using JAN_JUN_TOTAL.
- ColumnTransformer for handling numerical and categorical features.
- Pipeline for combining preprocessing and model training.
- Chronological train-test splitting based on year.

## 🤖 Machine Learning Models

This project implements 10 machine learning algorithms.

### 1. Regression Models

Regression models predict continuous annual rainfall values.

| No. | Algorithm |
|---|---|
| 1 | Simple Linear Regression |
| 2 | Multiple Linear Regression |
| 3 | Polynomial Regression |
| 4 | Decision Tree Regressor |
| 5 | Support Vector Regression (SVR) |

### 2. Classification Models

Classification models categorize annual rainfall into three categories:

- Low Rainfall
- Medium Rainfall
- High Rainfall

| No. | Algorithm |
|---|---|
| 6 | Logistic Regression |
| 7 | Decision Tree Classifier |
| 8 | Linear Support Vector Machine |
| 9 | RBF Support Vector Machine |
| 10 | Polynomial Support Vector Machine |

## 📊 Model Evaluation Metrics

### Regression Evaluation

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Classification Evaluation

- Accuracy
- Precision
- Recall
- F1-Score
- Log Loss
- ROC-AUC Score

## 📈 Key Results

### Regression Performance

| Model | RMSE | R² Score |
|---|---:|---:|
| Simple Linear Regression | 847.94 | 0.0088 |
| Multiple Linear Regression | 247.80 | 0.9154 |
| Polynomial Regression | 387.12 | 0.7934 |
| Decision Tree Regressor | 319.87 | 0.8590 |
| Support Vector Regression | 470.91 | 0.6943 |

**Best Regression Model:** Multiple Linear Regression

- R² Score: 0.9154
- RMSE: 247.80

### Classification Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 78.67% |
| Decision Tree Classifier | 72.85% |
| Linear SVM | 79.03% |
| RBF SVM | 80.24% |
| Polynomial SVM | 81.58% |

**Best Classification Model:** Polynomial SVM

- Accuracy: 81.58%

*Note: These results are from the project report. Classification AUC requires correction of class-label ordering before reporting numerical values.*

## 📉 Data Visualizations

The project generates the following visualizations:

1. Rainfall Correlation Heatmap
2. Average Annual Rainfall Trend
3. Annual Rainfall Distribution Histogram
4. Monthly Rainfall Distribution Box Plot
5. Regression Model Comparison
6. Actual vs Predicted Rainfall
7. Regression Residual Plots
8. Classification Confusion Matrices
9. Classification Accuracy Comparison
10. Decision Tree Visualization

## ⚙️ Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/indian-rainfall-analysis-and-prediction.git
```

### Step 2: Navigate to the Project Folder

```bash
cd indian-rainfall-analysis-and-prediction
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 4: Add the Dataset

Place `Sub_Division_IMD_2017.csv` in the project directory.

### Step 5: Run the Python Program

```bash
python "INT234 CODE.py"
```

The program will execute data preprocessing, train all 10 machine learning models, display evaluation metrics, compare model performance and generate visualizations.

## 📁 Project Structure

```text
indian-rainfall-analysis-and-prediction/
│
├── INT234 CODE.py
│
├── Sub_Division_IMD_2017.csv
│
├── Report INT234 CA2.docx
│
├── README.md
│
└── requirements.txt
```

## 🔮 Future Scope

- Integrate real-time weather APIs.
- Include temperature, humidity and wind speed.
- Implement Random Forest and XGBoost.
- Explore deep learning models such as LSTM.
- Develop an interactive dashboard using Streamlit.
- Implement advanced time-series forecasting techniques.

## 👨‍💻 Author

**Aniketh Reddy Anksapuram**

B.Tech – Computer Science and Engineering

Lovely Professional University

**Project:** Indian Rainfall Analysis and Prediction

**Course Code:** INT234

- GitHub: [Your GitHub Profile](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn Profile](YOUR_LINKEDIN_URL)

## 📚 References

- [India Meteorological Department](https://mausam.imd.gov.in/)
- [Python Documentation](https://www.python.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

⭐ If you find this project interesting, consider giving this repository a star!

**Made with Python and Machine Learning.**
