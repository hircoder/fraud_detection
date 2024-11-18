# fraud_monitoring_system.py

from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve, 
    auc,
    confusion_matrix,
    classification_report
)
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import logging
from pathlib import Path
from dataclasses import dataclass
import yaml
from abc import ABC, abstractmethod

# Type aliases
MetricsDict = Dict[str, Union[float, datetime]]
TransactionData = pd.DataFrame
AlertData = Dict[str, Any]

@dataclass
class MonitoringConfig:
    """Configuration for the monitoring system."""
    alert_threshold: float
    monitoring_window: str
    email_recipients: List[str]
    alert_rules: Dict[str, Any]
    visualization_settings: Dict[str, Any]

class AlertStrategy(ABC):
    """Abstract base class for alert strategies."""
    @abstractmethod
    def should_alert(self, data: Any) -> bool:
        pass

    @abstractmethod
    def generate_alert_data(self, data: Any) -> AlertData:
        pass

class HighRiskTransactionAlert(AlertStrategy):
    """Alert strategy for high-risk transactions."""
    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_alert(self, data: TransactionData) -> bool:
        return (data['fraud_probability'] >= self.threshold).any()

    def generate_alert_data(self, data: TransactionData) -> AlertData:
        high_risk = data[data['fraud_probability'] >= self.threshold]
        return {
            'num_transactions': len(high_risk),
            'total_amount': high_risk['amount'].sum(),
            'transactions': high_risk[['payment_id', 'amount', 'fraud_probability']].to_dict('records')
        }

class FraudMonitoringSystem:
    """
    Advanced fraud monitoring system with real-time pattern detection and alerting.
    
    This system provides:
    - Real-time transaction monitoring
    - Pattern detection and anomaly identification
    - Performance metric tracking
    - Automated alerting
    - Visualization of trends
    - Comprehensive reporting
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the fraud monitoring system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.metrics_history: List[MetricsDict] = []
        self.alert_strategies: List[AlertStrategy] = [
            HighRiskTransactionAlert(self.config.alert_threshold)
        ]
        self._setup_logging()
        self._setup_output_directories()

    def _load_config(self, config_path: Union[str, Path]) -> MonitoringConfig:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return MonitoringConfig(**config_dict)

    def _setup_logging(self) -> None:
        """Configure logging with rotation and formatting."""
        log_filename = f'logs/fraud_monitoring_{datetime.now().strftime("%Y%m%d")}.log'
        Path(log_filename).parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        self.logger = logging.getLogger('FraudMonitor')

    def _setup_output_directories(self) -> None:
        """Create necessary output directories."""
        directories = ['outputs', 'figures', 'reports', 'alerts']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)

    def calculate_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> MetricsDict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary containing various performance metrics
        
        Raises:
            ValueError: If input arrays have different lengths
        """
        try:
            if len(y_true) != len(y_pred) or len(y_true) != len(y_prob):
                raise ValueError("Input arrays must have the same length")

            # Basic metrics from confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate derived metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # ROC and PR curves
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            
            metrics: MetricsDict = {
                'timestamp': datetime.now(),
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'auc_roc': auc(fpr, tpr),
                'auc_pr': auc(recall_curve, precision_curve),
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
            
            self.metrics_history.append(metrics)
            self.logger.info(f"Performance metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def monitor_real_time_patterns(
        self,
        transactions: TransactionData,
        window_size: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Monitor real-time transaction patterns and detect anomalies.
        
        Args:
            transactions: DataFrame of recent transactions
            window_size: Time window for pattern analysis (default from config)
            
        Returns:
            Dictionary containing detected patterns and anomalies
        """
        try:
            window = window_size or self.config.monitoring_window
            transactions = transactions.copy()
            
            # Ensure proper datetime index
            transactions['timestamp'] = pd.to_datetime(transactions['adjusted_pmt_created_at'])
            transactions.set_index('timestamp', inplace=True)
            
            # Calculate various patterns
            patterns = {
                'transaction_velocity': self._calculate_velocity_patterns(transactions),
                'amount_patterns': self._analyze_amount_patterns(transactions),
                'risk_distribution': self._analyze_risk_distribution(transactions),
                'temporal_patterns': self._analyze_temporal_patterns(transactions, window)
            }
            
            self.logger.info("Real-time patterns analyzed successfully")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error monitoring patterns: {str(e)}")
            raise

    def _calculate_velocity_patterns(self, transactions: TransactionData) -> Dict[str, Any]:
        """Calculate transaction velocity patterns."""
        return {
            'per_ip': transactions.groupby('hashed_ip').size().describe().to_dict(),
            'per_consumer': transactions.groupby('hashed_consumer_id').size().describe().to_dict(),
            'hourly_volume': transactions.resample('H').size().describe().to_dict()
        }

    def _analyze_amount_patterns(self, transactions: TransactionData) -> Dict[str, Any]:
        """Analyze transaction amount patterns."""
        return {
            'statistics': transactions['amount'].describe().to_dict(),
            'anomalies': self._detect_amount_anomalies(transactions)
        }

    def _analyze_risk_distribution(self, transactions: TransactionData) -> Dict[str, Any]:
        """Analyze risk score distribution and patterns."""
        return {
            'high_risk_ratio': (transactions['fraud_probability'] >= self.config.alert_threshold).mean(),
            'risk_percentiles': transactions['fraud_probability'].describe().to_dict(),
            'risk_by_merchant': transactions.groupby('merchant_name')['fraud_probability'].mean().to_dict()
        }

    def _analyze_temporal_patterns(
        self,
        transactions: TransactionData,
        window: str
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in transactions."""
        return {
            'hourly_pattern': transactions.groupby(transactions.index.hour).size().to_dict(),
            'daily_pattern': transactions.groupby(transactions.index.dayofweek).size().to_dict(),
            'moving_averages': {
                'amount': transactions['amount'].rolling(window).mean().describe().to_dict(),
                'risk_score': transactions['fraud_probability'].rolling(window).mean().describe().to_dict()
            }
        }

    def generate_daily_report(
        self,
        transactions: TransactionData,
        metrics: MetricsDict
    ) -> str:
        """
        Generate comprehensive HTML daily report.
        
        Args:
            transactions: Day's transaction data
            metrics: Performance metrics
            
        Returns:
            HTML formatted report string
        """
        try:
            # Generate summary statistics
            summary_stats = self._calculate_summary_statistics(transactions)
            
            # Generate HTML report
            report = self._create_html_report(metrics, summary_stats)
            
            # Save report
            report_path = f'reports/daily_report_{datetime.now().strftime("%Y%m%d")}.html'
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Daily report generated: {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise

    def check_alerts(self, transactions: TransactionData) -> None:
        """
        Check for conditions that require alerts.
        
        Args:
            transactions: Current transaction data
        """
        try:
            for strategy in self.alert_strategies:
                if strategy.should_alert(transactions):
                    alert_data = strategy.generate_alert_data(transactions)
                    self.send_alert('high_risk_transaction', alert_data)
                    
        except Exception as e:
            self.logger.error(f"Error checking alerts: {str(e)}")
            raise

    def send_alert(self, alert_type: str, alert_data: AlertData) -> None:
        """
        Send alert email with detailed information.
        
        Args:
            alert_type: Type of alert
            alert_data: Alert details
        """
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f'Fraud Alert: {alert_type}'
            
            # Create detailed alert body
            body = self._create_alert_body(alert_type, alert_data)
            msg.attach(MIMEText(body, 'html'))
            
            # Save alert for record-keeping
            self._save_alert(alert_type, alert_data)
            
            # Send email (implementation depends on email server configuration)
            self._send_email(msg, self.config.email_recipients)
            
            self.logger.info(f"Alert sent: {alert_type}")
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")
            raise

    def visualize_trends(
        self,
        transactions: TransactionData,
        save_path: Optional[str] = None
    ) -> None:
        """
        Generate comprehensive visualizations of fraud detection trends.
        
        Args:
            transactions: Transaction data
            save_path: Optional path to save visualizations
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))
            
            # Plot various trends
            self._plot_risk_distribution(transactions, fig, 321)
            self._plot_amount_vs_risk(transactions, fig, 322)
            self._plot_temporal_patterns(transactions, fig, 323)
            self._plot_metric_trends(fig, 324)
            self._plot_volume_patterns(transactions, fig, 325)
            self._plot_merchant_risk(transactions, fig, 326)
            
            # Save or display
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
            self.logger.info("Trends visualization generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            raise

    def run_monitoring_cycle(
        self,
        transactions: TransactionData,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> None:
        """
        Run complete monitoring cycle including metrics, patterns, and alerts.
        
        Args:
            transactions: Current transaction data
            y_true: True fraud labels
            y_pred: Predicted fraud labels
            y_prob: Fraud probabilities
        """
        try:
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(y_true, y_pred, y_prob)
            
            # Monitor patterns
            patterns = self.monitor_real_time_patterns(transactions)
            
            # Generate report
            report = self.generate_daily_report(transactions, metrics)
            
            # Check for alerts
            self.check_alerts(transactions)
            
            # Generate visualizations
            self.visualize_trends(
                transactions,
                f'figures/trends_{datetime.now().strftime("%Y%m%d")}.png'
            )
            
            # Save monitoring cycle results
            self._save_cycle_results(metrics, patterns)
            
            self.logger.info("Monitoring cycle completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {str(e)}")
            raise

def main():
    """Main execution function."""
    try:
        # Initialize monitoring system
        monitor = FraudMonitoringSystem('config/monitoring_config.yml')
        
        # Load transactions
        transactions = pd.read_csv('data_scientist_fraud_20241009.csv')
        
        # Run monitoring cycle
        monitor.run_monitoring_cycle(
            transactions=transactions,
            y_true=transactions['fraud_flag'].fillna(0),
            y_pred=(transactions['fraud_probability'] >= monitor.config.alert_threshold).astype(int),
            y_prob=transactions['fraud_probability']
        )
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
