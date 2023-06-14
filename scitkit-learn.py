from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class IrisClassifier:
    def __init__(self):
        self.iris = load_iris()
        self.model = LogisticRegression(max_iter=200)

    def data_split(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.iris.data, 
            self.iris.target, 
            test_size=0.2, 
            random_state=42
        )
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    classifier = IrisClassifier()
    X_train, X_test, y_train, y_test = classifier.data_split()
    classifier.train_model(X_train, y_train)
    classifier.evaluate_model(X_test, y_test)
