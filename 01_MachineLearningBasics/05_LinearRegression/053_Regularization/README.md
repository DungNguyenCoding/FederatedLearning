# Regularization Techniques

Regularization is a technique used in machine learning to prevent overfitting by adding a penalty to the complexity of the model. It helps improve the generalization ability of the model by discouraging it from fitting noise in the training data.

## **What Is Regularization in Machine Learning?**

Regularization restricts a model to prevent overfitting by penalizing large coefficient values, with some techniques shrinking coefficients to zero. When a model suffers from overfitting, we should control the model's complexity. Technically, regularization avoids overfitting by adding a penalty to the model's loss function:

  $$
  Regularization = Loss function + Penalty
  $$

There are three commonly used regularization techniques to control the complexity of machine learning models:
- L1 regularization
- L2 regularization
- Elastic Net

### **1. L1 Regularization (Lasso Regression)**

Least Absolute Shrinkage and Selection Operator (lasso) regression is an alternative to ridge regression for regularizing linear models. It adds a penalty term to the cost function, known as `L1 regularization`, which encourages sparsity by shrinking some coefficients to exactly zero. This effectively ignores the least important features, emphasizing the model's most significant predictors.

- Adds the sum of absolute values of coefficients as a penalty:  
  $$
  L1 \text{ penalty} = \lambda \sum_{j=1}^{n} |\theta_j|
  $$

Here, $\lambda$ controls the strength of regularization, with larger values penalizing coefficients more, and $\theta_j$ represents the model's weights (coefficients).

By eliminating less important features, lasso regression performs automatic feature selection, simplifying the model and improving interpretability.

The objective function for Linear Models after applying lasso regression is:

$$
J(\theta) =\frac{1}{2m} \sum_{i=1}^m\left(\hat{y_i}-y_i\right) + \lambda \sum_{j=1}^{n} |\theta_j|
$$

### **2. L2 Regularization (Ridge Regression)**

A linear regression model that uses the `L2 regularization` technique is called ridge regression. Effectively, it adds a penalty term to the cost function, which reduces the magnitude of the model's weights (coefficients) without setting them to zero. This encourages the model to distribute influence more evenly across features, helping prevent overfitting while maintaining as much predictive power as possible.

- Adds the sum of squared values of coefficients as a penalty:  
  $$
  L2 \text{ penalty} = \lambda \sum_{j=1}^{n} \theta_j^2
  $$

Here, $\lambda$ controls the strength of regularization, and $\theta_j$ are the model's weights (coefficients). Increasing $\lambda$ applies stronger regularization, shrinking coefficients further, which can reduce overfitting but may lead to underfitting if $\lambda$ is too large. Conversely, when $\lambda$ is close to 0, the regularization term has little effect, and ridge regression behaves like ordinary linear regression.

Ridge regression helps strike a balance between bias and variance, improving the model's ability to generalize to unseen data by controlling the influence of less important features.

The objective function for Linear Models after applying lasso regression is:

$$
J(\theta) =\frac{1}{2m} \sum_{i=1}^m\left(\hat{y_i}-y_i\right) + \lambda \sum_{j=1}^{n} \theta_j^2
$$

### **3. Elastic Net Regularization**

The Elastic Net is a regularized regression technique combining ridge and lasso's regularization terms. The 
 parameter controls the combination ratio. When $\alpha = 1$, the L2 term will be eliminated, and when $\alpha = 0$, the L1 term will be removed.

- Combines L1 and L2 penalties:  
  $$
  \lambda_1 \sum_{j=1}^{p} |\theta_j| + \lambda_2 \sum_{j=1}^{p} \theta_j^2
  $$

Useful when working with correlated features. Although combining the penalties of lasso and ridge usually works better than only using one of the regularization techniques, adjusting two parameters, $\lambda$ and $\alpha$, is a little tricky.

Elastic Net Regression is a hybrid regularization technique that combines the power of both L1 and L2 regularization in linear regression objective:

$$
J(\theta) =\frac{1}{2m} \sum_{i=1}^m\left(\hat{y_i}-y_i\right) + \alpha \lambda \sum_{j=1}^{n} |\theta_j| + \frac{1}{2} (1 - \alpha) \lambda \sum_{j=1}^{n} \theta_j^2
$$

### **4. Dropout Regularization (for Neural Networks)**
- Randomly drops neurons during training to prevent co-adaptation of neurons.

### **5. Batch Normalization**
- Normalizes inputs to each layer to stabilize training and improve generalization.

### **6. Early Stopping**
- Stops training when validation loss starts increasing, preventing overfitting.

### **7. Data Augmentation**
- Expands the dataset artificially to make the model more robust.


## Reference

[Linear Regression in Machine learning by **GeeksforGeeks**](https://www.geeksforgeeks.org/ml-linear-regression/)