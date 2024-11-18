# Fraud Detection System

A machine learning algorithm for detecting fraudulent transactions in e-commerce payment data. The system implements feature engineering, real-time monitoring, and automated alerting capabilities.

## Project Structure

```
fraud_detection/
├── src/
│   ├── fraud_detection_system.py     # Main implementation
│   ├── fraud_detection_pipeline.py   # Scikit-learn pipeline implementation
│   └── fraud_monitoring_system.py    # Real-time monitoring implementation
├── notebooks/
│   └── Fraud_Detection_Notebook.ipynb               # Exploratory analysis
├── outputs/
│   ├── figures/                     # Visualization outputs
│   ├── models/                      # Saved models
│   └── logs/                        # System logs
└── README.md
```

## Features

### Core Functionality
- Feature engineering with proper temporal handling
- Automated alert system
- Comprehensive performance metrics

### Key Features
1. **Time-Based Feature Engineering**
   - Transaction velocity analysis
   - Rolling statistics
   - Temporal pattern detection

2. **Risk Scoring**
   - Multi-factor risk assessment
   - Account age risk

3. **Model Pipeline**
   - Scikit-learn preprocessing pipeline
   - XGBoost classifier
   - Cross-validation framework
   - Performance evaluation

4. **Monitoring System**
   - Real-time pattern detection
   - Automated alerting
   - Business impact analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/hircoder/fraud_detection.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dependencies
```
pandas>=1.2.0
numpy>=1.19.0
scikit-learn>=0.24.0
xgboost>=1.5.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## Usage

### Basic Usage
```python
from fraud_detection_system import FraudDetectionSystem

# Initialize system
fraud_system = FraudDetectionSystem()

# Load and preprocess data
df = fraud_system.load_and_preprocess_data('data.csv')

# Train model
X_data = fraud_system.prepare_features(df)
model = fraud_system.train_model(X_data, df['fraud_flag'])

# Make predictions
predictions = fraud_system.predict_fraud_probability(X_data)
```

### Monitoring System
```python
from fraud_monitoring_system import FraudMonitoringSystem

# Initialize monitoring
monitor = FraudMonitoringSystem('config/monitoring_config.yml')

# Run monitoring cycle
monitor.run_monitoring_cycle(
    transactions=transactions,
    y_true=true_labels,
    y_pred=predictions,
    y_prob=probabilities
)
```

## Configuration

Example monitoring configuration:
```yaml
alert_threshold: 0.8
monitoring_window: '1H'
email_recipients:
  - fraud_team@company.com
alert_rules:
  high_risk:
    threshold: 0.9
    min_amount: 10000
```

## Performance Metrics

The system tracks multiple performance metrics:
- ROC-AUC score
- Precision-Recall curve
- F1 score
- Custom business metrics

## Documentation

Detailed documentation is available:
- Technical implementation details
- Configuration guide

