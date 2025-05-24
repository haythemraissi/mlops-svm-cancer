import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(["Unnamed: 32", "id"], axis=1, inplace=True, errors='ignore')
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data("data/Dataset.csv")
    import joblib
    joblib.dump(scaler, 'scaler.pkl')