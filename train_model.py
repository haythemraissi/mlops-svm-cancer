import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from data_preparation import load_and_prepare_data
import joblib

# Configurer MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Lancez le serveur MLflow avant
mlflow.set_experiment("Breast_Cancer_Classification")

# Charger et préparer les données
X_train, X_test, y_train, y_test, scaler = load_and_prepare_data("data/Dataset.csv")

with mlflow.start_run():
    # Entraîner le modèle SVM
    svm_model = SVC(random_state=101)
    svm_model.fit(X_train, y_train)

    # Évaluer le modèle
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Logger les paramètres et métriques
    mlflow.log_param("random_state", 101)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metrics({
        "precision_B": report["B"]["precision"],
        "recall_B": report["B"]["recall"],
        "f1_B": report["B"]["f1-score"],
        "precision_M": report["M"]["precision"],
        "recall_M": report["M"]["recall"],
        "f1_M": report["M"]["f1-score"]
    })

    # Logger le modèle et le scaler
    mlflow.sklearn.log_model(svm_model, "svm_model")
    mlflow.log_artifact("scaler.pkl")

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Sauvegarder le modèle localement pour l'API
    joblib.dump(svm_model, 'svm_model.pkl')