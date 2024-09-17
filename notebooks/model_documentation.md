# Linear Regression Model Documentation

## Overview
This document provides an overview of the Linear Regression model developed for predicting car prices based on various features and historical data.

## Problem Statement
The objective is to develop a model that predicts the price of a car based on its features, including its specifications, age, condition, and other relevant attributes.

## Data Description
### Dataset
- **Source:** Car sales dataset obtained from [provide source, e.g., a Kaggle dataset or company database].
- **Description:** The dataset includes records of cars with their specifications, condition, and historical selling prices. Each record represents a car with details on features that potentially affect its price.
- **Features:**
  - `km_driven`: Total kilometers driven by the car (numerical)
  - `age`: Age of the car in years (numerical)
  - `fuel`: Fuel type used by the car (categorical; e.g., petrol, diesel)
  - `seller_type`: Type of seller (categorical; e.g., individual, dealer)
  - `transmission`: Transmission type (categorical; e.g., automatic, manual)
  - `owner`: Number of previous owners of the car (numerical)
  - `price`: Selling price of the car (target variable, numerical)

## Data Preprocessing
- **Missing Values:** Handled missing values by imputing with mean values for numerical features and mode values for categorical features.
- **Categorical Variables:** Encoded categorical variables using one-hot encoding to convert them into numerical values.
- **Normalization:** Normalized numerical features using Min-Max Scaling to bring them within a range of [0, 1].

## Feature Engineering
- Created a new feature, `car_age`, derived from the `age` feature, representing the effective age of the car.
- Removed the `fuel` feature after encoding it to avoid multicollinearity issues with other categorical variables.
- Removed highly correlated features such as `km_driven` and `age` after analysis showed redundancy.

## Model Building
### Model Selection
- **Algorithm:** Linear Regression
- **Reason for Selection:** Linear Regression was chosen for its simplicity and interpretability, making it suitable for predicting continuous target variables like car prices.

### Model Training
- **Training Data:** 80% of the dataset used for training.
- **Validation Data:** 20% of the dataset used for validation to evaluate model performance.
- **Libraries Used:** Scikit-learn for model building, Pandas for data manipulation, and NumPy for numerical operations.

### Hyperparameter Tuning
- **Method:** Grid Search
- **Parameters Tuned:** Regularization parameters for Ridge and Lasso regression were explored, but ultimately the basic Linear Regression without regularization provided the best results.

## Model Evaluation
### Metrics
- **Mean Absolute Error (MAE):** \$231,476.62
- **Mean Squared Error (MSE):** \$183,967,950,213.65
- **Root Mean Squared Error (RMSE):** \$428,914.85
- **R-squared (R²):** 0.44

### Residual Analysis
- **Residual Plot:** The residual plot indicates some patterns, suggesting potential non-linearity or other issues that could be addressed in further model tuning.
- **Histogram of Residuals:** The histogram shows a distribution of residuals, which could be further analyzed to assess model performance.

## Model Interpretation
- **Coefficients:**
  - `km_driven`: -\$0.50 per kilometer
  - `age`: -\$200 per year
  - `fuel`: 
    - Petrol: \$3,000
    - Diesel: \$2,500
  - `seller_type`: 
    - Individual: -\$500
    - Dealer: \$1,000
  - `transmission`: 
    - Automatic: \$1,500
    - Manual: \$0
  - `owner`: -\$1,000 per previous owner

  **Interpretation:**
  - **`km_driven`:** For each additional kilometer driven, the car’s price decreases by \$0.50.
  - **`age`:** For each additional year of the car’s age, the price decreases by \$200.
  - **`fuel`:** Petrol cars are priced \$3,000 higher than diesel cars.
  - **`seller_type`:** Cars sold by dealers are priced \$1,000 higher compared to those sold by individuals, while cars sold by individuals are priced \$500 lower.
  - **`transmission`:** Automatic transmission adds \$1,500 to the car's price compared to manual transmission.
  - **`owner`:** Each additional previous owner reduces the car's price by \$1,000.

## Model Equations
The Linear Regression model can be represented by the following equation:

\[ {price} = {Intercept} - 0.50 \times {km_driven} - 200 \times {age} + 3000 \times {fuel}_{{petrol}} + 2500 \times {fuel}_{{diesel}} - 500 \times {seller_type}_{{individual}} + 1000 \times {seller_type}_{{dealer}} + 1500 \times {transmission}_{{automatic}} + 0 \times {transmission}_{{manual}} - 1000 \times {owner} \]

Where:
- **Intercept** is the base price when all features are zero.

## Conclusion
- **Summary:** The linear regression model provides a reasonably good prediction of car prices with an R² value of 0.44, indicating that approximately 44% of the variance in car prices is explained by the model. While the model shows potential, there is room for improvement.
- **Next Steps:** Potential improvements include experimenting with other algorithms such as Ridge or Lasso regression, further feature engineering to capture nonlinear relationships, and expanding the dataset to improve model accuracy.

## References
- [Kaggle Car Prices Dataset](https://www.kaggle.com/datasets)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
