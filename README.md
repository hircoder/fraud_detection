# Fraud Detection System

A machine learning system for detecting and preventing fraudulent transactions in e-commerce payments, featuring real-time monitoring, automated alerting, and future prediction capabilities.

## Project Overview

### Dataset Characteristics
- Total Transactions: 13,239
- Fraudulent Transactions: 69 (0.52%)
- Time Period: April 26 - May 8, 2021
- Key Event: Suspicious activity detected on April 27th

### Core Capabilities
1. **Fraud Detection Engine**
   - Future prediction
   - Feature engineering
   - Real-time scoring
   - Performance validation

2. **Monitoring System**
   - Real-time pattern detection
   - Transaction velocity monitoring
   - Risk distribution analysis
   - Temporal pattern analysis

3. **Alert System**
   - Alert framework
   - Risk-based alerting
   - Performance monitoring
   - Business impact tracking

## Project Structure
```
fraud_detection/
├── src/
│   ├── future_fraud_detection.py    # Future prediction implementation
│   ├── FraudDetection_pipeline_main.py    # Base detection engine
│   ├── fraud_monitoring_system.py   # Real-time monitoring system
│   └── fraud_detection_system.py    # Base detection engine
│   └── fraud_detection_time_based.py # Basic temporal features
├── Notebook
│   ├── Fraud_Detection_Notebook.ipynb # Basic detection engine in Jupyter notebook style
│── monitoring_config.yml        # Monitoring configuration
│── alert_config.yml            # Alert system configuration, to be added
├── outputs/
│   ├── future_predictions/         # Future prediction results
│   │   ├── high_risk_transactions.csv
│   │   ├── summary_statistics.csv
│   │   └── performance_report.txt
│   └── monitoring/
│       ├── alerts/                 # Generated alerts
│       └── patterns/               # Detected patterns
├── figures/
│   ├── future_analysis/
│   │   ├── probability_distribution.png
│   │   ├── risk_level_analysis.png
│   │   └── temporal_analysis.png
│   └── monitoring/
│       ├── pattern_analysis.png
│       └── alert_distribution.png
├── docs/
│   ├── technical-summary.md
│   ├── Final_Report,md
│   ├── fraud-monitoring-alert-docs.md
│   ├── code-evolution.md
│   └── future-fraud-detection-docs.md
└── logs/
    └── fraud_detection.log
```

## Features

### 1. Future Fraud Detection
```python
# Initialize detection systems
base_system = FraudDetectionSystem()
future_detector = FutureFraudDetection(base_system, cutoff_date='2021-04-27')

# Generate predictions
future_predictions = future_detector.flag_future_transactions(df)
```

### 2. Feature Engineering
```python
# Time-based features
df['account_age_hours'] = (
    df['adjusted_pmt_created_at'] - 
    df['adjusted_acc_created_at']
).dt.total_seconds() / 3600

# Transaction velocity
df['tx_count_1H'] = df.groupby('hashed_consumer_id', group_keys=False)['payment_id'].apply(
    lambda x: x.shift().rolling('1H').count()
)
```

### 3. Real-time Monitoring
```python
# Initialize monitoring system
monitor = FraudMonitoringSystem(config_path='config/monitoring_config.yml')

# Run monitoring cycle
monitor.run_monitoring_cycle(
    transactions=transactions,
    y_true=labels,
    y_pred=predictions,
    y_prob=probabilities
)
```

### 4. Alert System
```python
class HighRiskTransactionAlert(AlertStrategy):
    def should_alert(self, data: TransactionData) -> bool:
        return (data['fraud_probability'] >= self.threshold).any()

    def generate_alert_data(self, data: TransactionData) -> AlertData:
        high_risk = data[data['fraud_probability'] >= self.threshold]
        return {
            'num_transactions': len(high_risk),
            'total_amount': high_risk['amount'].sum(),
            'transactions': high_risk[['payment_id', 'amount', 'fraud_probability']].to_dict('records')
        }
```

## Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/fraud_detection](https://github.com/hircoder/fraud_detection/).git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix

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
pyyaml>=5.4.0
```

## Configuration

### Monitoring Configuration
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

### Performance Targets
```yaml
performance_targets:
  precision: 0.95
  recall: 0.80
  f1_score: 0.85
  false_positive_rate_max: 0.05
  response_time_seconds: 1.0
```

## Usage Examples

### 1. Basic Fraud Detection
```python
from future_fraud_detection import FraudDetectionSystem

# Initialize and train model
detector = FraudDetectionSystem()
df = detector.load_and_preprocess_data('transaction_data.csv')
X_train_data = detector.prepare_features(train_data)
detector.train_model(X_train_data, train_data['fraud_flag'])
```

### 2. Monitoring Implementation
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

### 3. Alert Configuration
```python
# Define custom alert strategy
class VolumeAlert(AlertStrategy):
    def should_alert(self, data):
        return data['transaction_count'] > self.threshold

# Add to monitoring system
monitor.alert_strategies.append(VolumeAlert(threshold=1000))
```

## Performance Metrics

### Model Performance
```python

Final Metrics:
{
    'accuracy': 0.9653,    # Overall correct predictions
    'precision': 0.0622,   # True positive / Total predicted positive
    'recall': 1.0000,      # True positive / Total actual positive
    'f1': 0.1172,         # Harmonic mean of precision and recall
    'f2': 0.2491,         # Weighted F-score favoring recall
    'auc_roc': 0.9877,    # Area under ROC curve
    'auc_pr': 0.1606      # Area under Precision-Recall curve
}
```
High AUC-ROC (0.9877) indicates excellent discrimination ability
Perfect recall (1.0000) suggests we catch all fraud cases
Low precision (0.0622) indicates many false positives
F2 score (0.2491) shows recall-focused performance

#### Metric Selection by Objective
1. Minimize Financial Loss (High Precision)
   - High cost of investigating false alarms
   - Limited investigation resources
   - Focus on high-confidence fraud cases
      **Key Metrics: Precision, AUC-PR**
      **Trade-off: May miss some fraudulent transactions**

2. Maximum Fraud Detection (High Recall)
  - High cost of missed fraud
  - Sufficient resources for investigation
  - In case of strict requirements
    **Key Metrics: Recall, F2 Score
    Trade-off: More false positives to investigate**

3. Balanced Approach (F1 Score)
   - Equal cost of false positives and false negatives
   - Moderate investigation resources
     **Key Metrics: F1 Score, AUC-ROC
     Trade-off: Compromise between missed fraud and false alarms**

Precision-Recall Trade-off:
   - Higher precision → Lower recall
   - Higher recall → Lower precision
   - Business decides optimal balance

Therefore, given the final metrics reported, this model is suitable for:
   - High-stakes fraud detection where missing fraud is costly
   - Environments with sufficient investigation resources
   - Cases where false positives are less costly than false negatives

### Monitoring Metrics
- Transaction velocity patterns
- Risk distribution analysis
- Temporal pattern detection
- Business impact assessment

## Documentation

### Technical Documentation
- [Documentations of code for future fraud detection](future-fraud-detection-docs.md)
- [Documentations of monitoring and alert modules](fraud-monitoring-alert-docs.md)
- [Understaing the code deeply](code-evolution.md)
- [Brief technical summary of codes and implementations](technical-summary.md)
- [Report and answers for assigned tasks](Final_Report,md)

### Implementation Notes
1. Proper handling of temporal data
2. Prevention of data leakage
3. Robust error handling
4. Comprehensive logging system
5. Real-time monitoring capabilities
6. Alert system implementation


## Acknowledgments
- Built using scikit-learn and XGBoost: https://github.com/dmlc/xgboost/tree/master/demo/guide-python https://github.com/scikit-learn/scikit-learn/tree/main/examples
- Monitoring system inspired by production fraud detection systems
- Visualization powered by matplotlib and seaborn
- Alert system based on enterprise monitoring patterns
