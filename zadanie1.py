import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score

class PriorClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_, self.class_counts_ = np.unique(y, return_counts=True) #zapamiętywanie licznosci klas zbioru treningowego
        self.class_probabilities_ = self.class_counts_ / self.class_counts_.sum() #obliczam prawdopodobieństwo wystąpienia każdej klasy

    def predict(self, X):
        return np.random.choice(self.classes_, size=len(X), p=self.class_probabilities_) #losowanie etykiety zgodnie z rozkładem klas


X, y = make_classification(n_samples=500, weights=[0.8, 0.2]) #zbior syntetyczny

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y) #podział danych na zbiór treningowy i testowy

clf = PriorClassifier()
clf.fit(X_train, y_train) #trenowanie klasyfikatora

y_pred = clf.predict(X_test) #predykcja etykiet dla zbioru testowego

print("\nZadanie 1:")
print("Balanced Accuracy Score:", round(balanced_accuracy_score(y_test, y_pred), 3))
print("Accuracy Score:", round(accuracy_score(y_test, y_pred), 3))