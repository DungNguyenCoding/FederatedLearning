import mglearn as ml
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import requests
from io import StringIO

# URL of the dataset
url = "https://raw.githubusercontent.com/datatweets/tutorials/refs/heads/main/misc/boston_housing.csv"

# Fetch the data
response = requests.get(url)
if response.status_code == 200:
    # Convert to NumPy array
    dataset = genfromtxt(StringIO(response.text), delimiter=",")
    print("Success to download dataset!")
else:
    print("Failed to download dataset!")
    exit()

X = dataset[:,:-1]
y = dataset[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print("\n---------------------------------------------------\n")

# Train the Linear regression model
lr = LinearRegression().fit(X_train, y_train)
print("Train the Linear regression model:")
print(f"Linear Regression-Training set score: {lr.score(X_train, y_train):.2f}")
print(f"Linear Regression-Test set score: {lr.score(X_test, y_test):.2f}")
print("\n---------------------------------------------------\n")

# Applying L1 Regularization
a = 1.0
lasso = Lasso(alpha= a).fit(X_train, y_train)
print(f"Applying L1 Regularization (alpha= {a}):")
print(f"Lasso Regression-Training set score: {lasso.score(X_train, y_train):.2f}")
print(f"Lasso Regression-Test set score: {lasso.score(X_test, y_test):.2f}")
print(f"Number of features: {sum(lasso.coef_ != 0)}")
print("\n---------------------------------------------------\n")

a = 0.01
lasso = Lasso(alpha= a).fit(X_train, y_train)
print(f"Applying L1 Regularization (alpha= {a}):")
print(f"Lasso Regression-Training set score: {lasso.score(X_train, y_train):.2f}")
print(f"Lasso Regression-Test set score: {lasso.score(X_test, y_test):.2f}")
print(f"Number of features: {sum(lasso.coef_ != 0)}")
print("\n---------------------------------------------------\n")

# Applying L2 Regularization
ridge = Ridge(alpha=0.7).fit(X_train, y_train)
print("Applying L2 Regularization:")
print(f"Ridge Regression-Training set score: {ridge.score(X_train, y_train):.2f}")
print(f"Ridge Regression-Test set score: {ridge.score(X_test, y_test):.2f}")
print("\n---------------------------------------------------\n")

# Applying Elastic Net Regularization
elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.01).fit(X_train, y_train)
print("Applying Elastic Net Regularization:")
print(f"Elastic Net-Training set score: {elastic_net.score(X_train, y_train):.2f}")
print(f"Elastic Net-Test set score: {elastic_net.score(X_test, y_test):.2f}")