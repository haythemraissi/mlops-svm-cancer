import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_preparation import load_and_prepare_data

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Breast_Cancer_Classification")

X_train, X_test, y_train, y_test, scaler = load_and_prepare_data("data/Dataset.csv")

with mlflow.start_run():
    model = SVC(random_state=101)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "svm_model")
    mlflow.log_artifact("scaler.pkl")