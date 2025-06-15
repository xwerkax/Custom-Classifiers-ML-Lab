import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier


class MyKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = cdist(X, self.X_train)  # odległości euklidesowe między próbkami testowymi a treningowymi

        nearest_neighbors = np.argsort(distances, axis=1)[:, :self.k] # indeksy k najbliższych sąsiadów dla każdej próbki testowej

        y_pred = []
        for neighbors in nearest_neighbors:
            labels, counts = np.unique(self.y_train[neighbors], return_counts=True)
            majority_label = labels[np.argmax(counts)]
            y_pred.append(majority_label)

        return np.array(y_pred)

X, y = make_classification(n_samples=500, weights=[0.8, 0.2])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y) # #podział danych na zbiór treningowy i testowy

knn_custom = MyKNN(k=5)
knn_custom.fit(X_train, y_train)
y_pred_custom = knn_custom.predict(X_test)

knn_sklearn = KNeighborsClassifier(n_neighbors=5, algorithm='brute') # Użycie brute-force do porównania
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)

print("\nZadanie 2:")
print("Balanced Accuracy Score -> custom KNN:", round(balanced_accuracy_score(y_test, y_pred_custom), 3))
print("Balanced Accuracy Score -> sklearn KNN:", round(balanced_accuracy_score(y_test, y_pred_sklearn), 3))