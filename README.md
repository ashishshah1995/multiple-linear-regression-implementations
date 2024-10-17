# Multiple Linear Regression Implementations

This repository contains two implementations of multiple linear regression:
1. **From Scratch** using NumPy. A custom implementation from scratch using Python and NumPy.
2. **Using Scikit-Learn** for comparison. An implementation using scikit-learn's LinearRegression model.

## Project Overview

Multiple linear regression is a statistical technique that models the relationship between a dependent variable and multiple independent variables. The goal is to fit a linear equation to the observed data.

In this project, I have implemented the multiple linear regression algorithm manually and using the Scikit-Learn library to understand and compare both approaches.

### Dataset

The dataset used is the **Diabetes dataset** from Scikit-Learn, which consists of 442 samples and 10 feature variables. The target variable represents a quantitative measure of disease progression one year after baseline.

- **Features**: 10 continuous variables (age, sex, body mass index, average blood pressure, and six blood serum measurements).
- **Target**: A continuous variable representing the disease progression.

## Files
- **`multiple_linear_regression_from_scratch.ipynb`**: Contains the custom implementation of multiple linear regression from scratch using Python and NumPy.
- **`multiple_linear_regression_manual.ipynb`**: Contains the implementation of multiple linear regression using the `scikit-learn` library.

### Implementations

1. **Multiple Linear Regression from Scratch**:  
   In this approach, I manually calculate the regression coefficients using the closed-form solution derived from linear algebra, specifically the Normal Equation.
   In the `multiple_linear_regression_from_scratch.ipynb`, I build the multiple linear regression model using NumPy to perform matrix operations.

### Key Steps:
1. **Insert Bias Term**: A column of 1s is added to the feature matrix to account for the intercept term.
2. **Closed-Form Solution (Normal Equation)**: The coefficients (`betas`) are calculated using the formula:
   \[
   
   beta = (X^T X)^{-1} X^T y
   \]
   Where:
   - \(X\) is the matrix of input features with a bias term.
   - \(y\) is the target variable vector.
   - (beta\) represents the coefficients (including the intercept).

   
4. **Scikit-Learn Implementation**:  
   The Scikit-Learn library provides a built-in implementation of multiple linear regression. This version is also included for comparison.

   ### Metrics:
- After training the model, I evaluate its performance using the **R² score** (coefficient of determination), which provides an indication of how well the model predicts the target values.

  ## Implementation Using scikit-learn

In the `multiple_linear_regression_manual.ipynb`, I use the `LinearRegression` model from the `scikit-learn` library to fit the multiple linear regression model and make predictions.

### Key Steps:
1. **Model Fitting**: We use the `fit()` method of `LinearRegression` to train the model.
2. **Prediction**: We use the `predict()` method to make predictions on the test set.
3. **Evaluation**: The model’s performance is evaluated using the `r2_score()` function.


### Results

Both implementations predict the target variable using the features from the test set, and the results are evaluated using the R-squared metric.

### Usage

#### Running the Scratch Implementation

The scratch implementation is provided in the `multiple_linear_regression_from_scratch.ipynb` file.

#### Running the Scikit-Learn Implementation

To run the Scikit-Learn version, use the `multiple_linear_regression_manual.ipynb` file.

### How to Use

1. **Clone the Repository**:
   ```
   git clone https://github.com/ashishshah1995/multiple-linear-regression-implementations.git
   ```

2. **Install Dependencies**:
   This project uses `numpy` and `scikit-learn`. Install the necessary dependencies using pip:
   ```
   pip install numpy scikit-learn
   ```

3. **Run the Jupyter Notebooks**:
   Open the Jupyter notebooks (`multiple_linear_regression_from_scratch.ipynb` and `multiple_linear_regression_manual.ipynb`) to see the two implementations in action.

### Conclusion

The purpose of this project is to demonstrate the manual implementation of multiple linear regression, understand the underlying math, and compare it with the more abstracted Scikit-Learn implementation. Both approaches lead to similar results, but the Scikit-Learn version is more efficient and easier to use for practical applications.

