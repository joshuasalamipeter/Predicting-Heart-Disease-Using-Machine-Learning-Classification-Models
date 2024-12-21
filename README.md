# Predicting Heart Disease Using Machine Learning Classification Models

## Project Overview
- **Author:** Joshua Salami Peter
- **Completion Date:** December 12, 2024
- **Email:** jsalamipeter@gmail.com
- **LinkedIn:** [Joshua Salami Peter](https://www.linkedin.com/in/jsalamipeter/)

This project focuses on using machine learning to predict heart disease based on clinical and demographic data. By leveraging classification algorithms, the project aims to assist healthcare professionals in early detection and diagnosis of heart disease.

## Background
Heart disease, or cardiovascular disease, is a leading cause of death worldwide. Early detection is vital to improving patient outcomes. Traditional diagnostic methods are often time-consuming and prone to error. Machine learning offers a data-driven approach to enhance diagnostic accuracy and efficiency.

### Key Risk Factors:
- High blood pressure
- High cholesterol
- Obesity
- Diabetes
- Smoking and excessive alcohol consumption

### Preventive Measures:
- Healthy diet
- Regular exercise
- Avoidance of tobacco and excessive alcohol
- Overall lifestyle changes

## Objectives
1. Preprocess the dataset (handle missing values, outliers, and feature scaling).
2. Perform exploratory data analysis (EDA) and visualizations.
3. Identify significant features using feature selection techniques.
4. Train and evaluate multiple machine learning models.
5. Compare model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
6. Recommend the best-performing model for clinical application.

## Methodology

### Data Collection and Preprocessing
- **Dataset:** Heart Disease UCI Dataset ([link](https://archive.ics.uci.edu/ml/datasets/Heart+Disease))
- Steps:
  - Handle missing values using imputation.
  - Standardize numerical features.
  - Encode categorical variables.
  - Address class imbalance with oversampling.

### Exploratory Data Analysis (EDA)
- Visualize data distributions and relationships between features.
- Use correlation heatmaps to identify collinear features.

### Feature Selection
- Use Random Forest-based feature importance to select key predictors.

### Machine Learning Models
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine

### Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Use cross-validation to ensure generalizability.
- Perform hyperparameter tuning using GridSearchCV.

## Results
### Before Oversampling
- Best model: Support Vector Machine (SVM)
- **Accuracy:** 86.88%
- **Precision:** 90.00%
- **Recall:** 84.38%
- **F1-Score:** 87.10%
- **ROC-AUC:** 94.07%

### After Oversampling and Hyperparameter Tuning
- Best model: Random Forest
- **Accuracy:** 88.52%
- **Precision:** 90.32%
- **Recall:** 87.50%
- **F1-Score:** 88.89%
- **ROC-AUC:** 95.15%

## Conclusion
Random Forest emerged as the best-performing model due to its robustness and excellent classification metrics. The model balances precision and recall, making it a reliable tool for heart disease prediction.

### Recommendations
1. Refine the model further through advanced feature engineering and selection.
2. Perform feature selection to improve model performance.
3. Explore different techniques to enhance the model's generalization.
4. Improve robustness for deployment in real-world clinical settings.

## Tools and Technologies
- **Programming Language:** Python
- **Libraries:**
  - Numpy, Pandas, Matplotlib, Seaborn
  - Scikit-learn 
  - Imbalanced-learn (RandomOversampler)
- **Environment:** Jupyter Notebook

## Contact
- **Email:** jsalamipeter@gmail.com
- **Phone:** +2349097265528
- **LinkedIn:** [Joshua Salami Peter](https://www.linkedin.com/in/jsalamipeter/)

