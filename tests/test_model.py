import pytest
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_preparation import load_and_prepare_data

def test_model_accuracy():
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data("data/Dataset.csv")
    model = SVC(random_state=101)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert accuracy_score(y_test, y_pred) > 0.9