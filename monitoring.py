from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
import pandas as pd

# Charger les données de référence (entraînement) et de production
reference_data = pd.read_csv('data/Dataset.csv').drop(['diagnosis', 'Unnamed: 32', 'id'], axis=1, errors='ignore')
production_data = pd.read_csv('data/production_data.csv')  # Remplacez par vos données de production

# Générer un rapport de dérive et de performance
report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
report.run(reference_data=reference_data, current_data=production_data)
report.save_html("drift_report.html")