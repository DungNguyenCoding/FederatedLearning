from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate model
def evaluate_model(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)
    return train_accuracy, test_accuracy

# Evaluate models with different k values
k_1 = evaluate_model(1)
k_5 = evaluate_model(5)
k_100 = evaluate_model(100)

print(f"K=1: Train Accuracy: {k_1[0]:.4f}, Test Accuracy: {k_1[1]:.4f}")
print(f"K=5: Train Accuracy: {k_5[0]:.4f}, Test Accuracy: {k_5[1]:.4f}")
print(f"K=100: Train Accuracy: {k_100[0]:.4f}, Test Accuracy: {k_100[1]:.4f}")