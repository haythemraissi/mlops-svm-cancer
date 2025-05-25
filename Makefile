```makefile
# Makefile for Breast Cancer SVM Model
# Project: Breast Cancer Diagnosis Classification
# ESPRIM Innovation Project, May 2025

# Variables
PYTHON = python3
PIP = pip3
VENV = venv
VENV_ACTIVATE = . $(VENV)/bin/activate
REQUIREMENTS = requirements.txt
MODEL_DIR = model
FIGURES_DIR = figures
DATA_DIR = data
DATASET = $(DATA_DIR)/Dataset.csv
MODEL_FILE = $(MODEL_DIR)/svm_model.pkl
SCALER_FILE = $(MODEL_DIR)/scaler.pkl
TRAIN_SCRIPT = $(MODEL_DIR)/train_svm.py
PREDICT_SCRIPT = $(MODEL_DIR)/predict_svm.py
FIGURES = $(FIGURES_DIR)/confusion_matrix.png

# Default target
.PHONY: all
all: setup train visualize

# Create virtual environment and install dependencies
.PHONY: setup
setup: $(VENV)/bin/activate
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(VENV_ACTIVATE) && $(PIP) install --upgrade pip
	$(VENV_ACTIVATE) && $(PIP) install -r $(REQUIREMENTS)

# Preprocess dataset (checks and prepares data)
.PHONY: preprocess
preprocess: $(VENV)/bin/activate $(DATASET)
	$(VENV_ACTIVATE) && $(PYTHON) -c "import pandas as pd; df = pd.read_csv('$(DATASET)'); df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True); print('Dataset preprocessed:', df.shape)"

# Train the SVM model
.PHONY: train
train: $(MODEL_FILE) $(SCALER_FILE)
$(MODEL_FILE) $(SCALER_FILE): $(VENV)/bin/activate $(TRAIN_SCRIPT) $(DATASET)
	$(VENV_ACTIVATE) && $(PYTHON) $(TRAIN_SCRIPT)

# Run inference on a sample patient
.PHONY: predict
predict: $(VENV)/bin/activate $(MODEL_FILE) $(SCALER_FILE)
	$(VENV_ACTIVATE) && $(PYTHON) $(PREDICT_SCRIPT) --patient_data "[15.0, 20.0, 100.0, 700.0, 0.1, 0.2, 0.15, 0.08, 0.2, 0.05, 0.5, 1.0, 3.0, 50.0, 0.01, 0.03, 0.02, 0.01, 0.02, 0.004, 18.0, 25.0, 120.0, 800.0, 0.14, 0.25, 0.2, 0.1, 0.3, 0.08]"

# Generate confusion matrix visualization
.PHONY: visualize
visualize: $(FIGURES)
$(FIGURES): $(VENV)/bin/activate $(MODEL_FILE) $(DATASET)
	$(VENV_ACTIVATE) && $(PYTHON) $(TRAIN_SCRIPT)

# Clean generated files
.PHONY: clean
clean:
	rm -rf $(MODEL_DIR)/*.pkl
	rm -rf $(FIGURES_DIR)/*.png
	rm -rf $(VENV)
	rm -rf __pycache__

# Full clean (including dataset, use with caution)
.PHONY: clean-all
clean-all: clean
	rm -rf $(DATA_DIR)/*

# Help message
.PHONY: help
help:
	@echo "Makefile for Breast Cancer SVM Model"
	@echo "Usage:"
	@echo "  make all        : Setup, train, and visualize"
	@echo "  make setup      : Create virtual environment and install dependencies"
	@echo "  make preprocess : Preprocess dataset (checks data)"
	@echo "  make train      : Train the SVM model"
	@echo "  make predict    : Run inference on sample patient data"
	@echo "  make visualize  : Generate confusion matrix"
	@echo "  make clean      : Remove generated model and figures"
	@echo "  make clean-all  : Remove all generated files and dataset"
	@echo "  make help       : Show this help message"
```