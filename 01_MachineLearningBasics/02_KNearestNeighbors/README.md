# K-Nearest Neighbors (KNN)

## Concepts

The `K-Nearest Neighbors` algorithm:
- For an unseen data point, the algorithm calculates the distance between that point andall the observations across all features in the training dataset.
- It sorts those distances in ascending order.
- It selects K observations with the smallest distances from the above step. These Kobservations are the K-nearest neighbors of that unseen data point.
- It calculates which label of those neighbors is the most common, and assigns that label tothe unseen data point.

A `distance metric` calculates the distance between two observations. 

One of the most common distance metrics is the Euclidean distance. For $n$ features, thisdistance can be calculated as:
$$ d(x, y) = \sqrt{\sum (x_i - y_i)^2} $$
Where, for each $i$ in $\{1,...,n\}$:
- $x_i$ is the value for a feature for one observation
- $y_i$ is the value for the same feature for another observation

K-nearest neighbors does not technically have a "training phase". The model classifies every newinput by comparing it to its neighbors. Those neighbors are data points from the training set. 

`Accuracy` of a model can be calculated as the percentage of correct predictions it makes out ofall predictions.

`Feature Engineering` is the process of transforming features so that they can be effectivelyused to train models and yield better performance. For example:
- `One-hot encoding` encodes categorical columns as numerical values
- `Min-max Scaling` or `Min-max Normalization` scales the values of a feature into therange $[0, 1]$.
Formula for min-max scaling:
$$X_{\text{scaled}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$
Where $X$ is the original value of the feature

## Syntax

Splitting a DataFrame into a training and test set

```python
train_df = banking_df.sample(frac=0.85, random_state=417)
test_df = banking_df.drop(train_df.index)
```

Calculating the Euclidean distance between two observations with just one feature

```python
abs(X_train[feature] - test_input[feature])
```

Calculating the accuracy of a model

```python
(X_test["predicted_y"] == y_test).value_counts(normalize=True)[0]*100
```

Creating dummy variables

```python
pd.get_dummies(data = banking_df_copy, columns = ["marital"], drop_first = True)
```

Calculating the Euclidean distance between two observations with multiple features

```python
distance = 0
for feature in features:
    distance += (X_train[feature] - test_input[feature])**2
    X_train["distance"] = (distance)**0.5
```

## Reference

[Machine Learning in Python by **Dataquest Labs, Inc**](https://app.dataquest.io/learning/path/machine-learning-in-python-skill/)