# Used Car Price Prediction

## Project Overview
This project predicts the resale price of used cars using machine learning techniques. The goal is to build a robust regression model that can assist buyers, sellers in estimating fair market prices.

The dataset used in this project is [US Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data/data) (Craigslist/eBay, 425,000+ rows), which contains detailed information about car listings such as year, brand, model, mileage, fuel type, transmission and condition.

## Objectives
* Perform data cleaning & preprocessing on a large real-world dataset.
* Handle missing values, outliers and categorical features.
* Extract meaningful regression models and evaluate their performance.
* Compare multiple regression models and evaluate their performance
* Build an interpretable, production-ready pipeline.

## Dataset
* Source: [Kaggle - Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data/data).
* Size : ~ 425,000 rows
* Features: size, cylinders, condition, drive, type, manufacturer, model, odometer, fuel, transmission, etc.
* Target: price

## Tech Stack
* Programming Language: Python (3.10.14)
* Libraries:
  *  Data Processing: Pandas, Numpy
  *  Visualization: Seaborn, Matplotlib
  *  ML Models: Scikit-learn, XGBoost, RandomForest, LightGBM

## Key Steps
**1. Data Preprocessing**
   *  Duplicate records removal
   *  Handling missing values (imputation strategies)
   *  Outlier Detection using Boxplots and Handling using IQR Method
   *  Encoding categorical variables (Label Encoding)
   *  Feature Scaling (StandardScalar)
**2. Exploratory Data Analysis (EDA)**
   * Price Distribution
   * Fuel Type Distribution, etc.
**3. Selecting Features based on p-values from Ordinary-Least-Squares Method**
**4. Model Building & Evaluation**
     * Spliting Data into Train,Test datasets
     * Models Implemented
       *  Linear Regression
       *  Ridge
       *  Lasso
       *  ElaticNet
       *  RandomForest
       *  GradientBoosting
       *  XGBoost
       *  LightGBM
     * Metrics Considered: **R² score**, **RMSE**
  **5. Hyperparameter tuning using RandomizedSearchCV on Tree-Based Regressors (RandomForest, GradientBoosting, XGB, LightGBM**

## Results
* Best-performing model:
* Acheived R² ~ and RMSE ~
* Important Features: 
       
