# Used Car Price Prediction

## Project Overview
This project predicts the resale price of used cars using machine learning techniques. The goal is to build a robust regression model that can assist buyers, sellers in estimating fair market prices.

The dataset used in this project is **US Used Cars Dataset (Craigslist, 425,000+ rows)**, which contains detailed information about car listings such as year, brand, model, mileage, fuel type, transmission and condition.

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

- **Data Preprocessing**  
  - Duplicate records removal  
  - Handling missing values (imputation strategies)  
  - Outlier Detection using Boxplots and Handling using IQR Method  
  - Encoding categorical variables (Label Encoding)  
  - Feature Scaling (StandardScaler)  

- **Exploratory Data Analysis (EDA)**  
  - Price Distribution  
  - Fuel Type Distribution, etc.  

- **Feature Selection**  
  - Selecting Features based on p-values from Ordinary-Least-Squares Method  

- **Model Building & Evaluation**  
  - Splitting Data into Train, Test datasets  
  - Models Implemented:
    - Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM  
  - Metrics Considered: **R² score**, **RMSE**
    | Model | R² Score | MSE |
|--------|------------------|-----------|
|   XGBoost | 0.6248 | 6.43e+07 |
|   LightGBM | 0.6174 | 6.56e+07 |
|   RandomForest| 0.5549 | 7.63e+07 |
|   GradientBoosting| 0.4994 | 8.58e+07 |
|   ElasticNet| 0.3487 | 1.11e+08 |
|   Lasso| 0.3486 | 1.11e+08 |
|   Ridge| 0.3486 | 1.11e+08 |
|   LinearRegression| 0.3486 | 1.11e+08|

- **Hyperparameter Tuning**  
  - RandomizedSearchCV on Tree-Based Regressors (RandomForest, GradientBoosting, XGB, LightGBM)
 
## Final Model Performance  
| Metric | Score |
|--------|------------------|
| R² Score | 0.7507 |
| MSE | 42,764,756.72 |
| RMSE | 6,539.48 |
- Model: RandomForestRegressor
- Parameters: n_estimators=100,min_samples_split=2,max_features='log2',max_depth=20
- Important Features: year, model, condition, cylinders, fuel ,odometer, title-status, transmission, drive, pain_color
       
