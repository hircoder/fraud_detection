# fraud_monitoring_system.py
#!/usr/bin/env python3
"""
Fraud Monitoring System

This module implements a comprehensive fraud monitoring system with real-time 
pattern detection, alerting, and visualization capabilities. It provides:
- Real-time monitoring
- Pattern detection and anomaly identification
- Performance metric tracking
- Automated alerting
- Visualization of trends
- Reporting

Author: Hose
Date: 2024-04-18

"""

# Standard library imports
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve, 
    auc,
    confusion_matrix,
    classification_report
)

# Type aliases for improved code readability
MetricsDict = Dict[str, Union[float, datetime]]
TransactionData = pd.DataFrame
AlertData = Dict[str, Any]

@dataclass
class MonitoringConfig:
    """
    Configuration class for the monitoring system.
    
    Attributes:
        alert_threshold (float): Threshold for triggering fraud alerts
        monitoring_window (str): Time window for pattern analysis (e.g., '1H', '24H')
        email_recipients (List[str]): List of email addresses for alerts
        alert_rules (Dict[str, Any]): Dictionary containing alert configuration rules
        visualization_settings (Dict[str, Any]): Settings for data visualization
    """
    alert_threshold: float
    monitoring_window: str
    email_recipients: List[str]
    alert_rules: Dict[str, Any]
    visualization_settings: Dict[str, Any]

class AlertStrategy(ABC):
    """
    Abstract base class for implementing different alert strategies.
    
    This class defines the interface for alert strategies. Subclasses must
    implement should_alert and generate_alert_data methods.
    """
    
    @abstractmethod
    def should_alert(self, data: Any) -> bool:
        """
        Determine if an alert should be triggered based on the data.
        
        Args:
            data: Data to be analyzed for alert conditions
            
        Returns:
            bool: True if alert should be triggered, False otherwise
        """
        pass

    @abstractmethod
    def generate_alert_data(self, data: Any) -> AlertData:
        """
        Generate alert data when alert conditions are met.
        
        Args:
            data: Data to generate alert information from
            
        Returns:
            AlertData: Dictionary containing alert details
        """
        pass

class HighRiskTransactionAlert(AlertStrategy):
    """
    Alert strategy for detecting high-risk transactions.
    
    This strategy triggers alerts when transactions exceed a risk threshold.
    
    Attributes:
        threshold (float): Risk threshold for triggering alerts
    """
    
    def __init__(self, threshold: float):
        """
        Initialize the high-risk transaction alert strategy.
        
        Args:
            threshold: Risk threshold for triggering alerts (0.0 to 1.0)
        """
        self.threshold = threshold

    def should_alert(self, data: TransactionData) -> bool:
        """
        Check if any transactions exceed the risk threshold.
        
        Args:
            data: DataFrame containing transaction data
            
        Returns:
            bool: True if any transactions exceed threshold
        """
        return (data['fraud_probability'] >= self.threshold).any()

    def generate_alert_data(self, data: TransactionData) -> AlertData:
        """
        Generate alert data for high-risk transactions.
        
        Args:
            data: DataFrame containing transaction data
            
        Returns:
            AlertData: Dictionary containing:
                - num_transactions: Number of high-risk transactions
                - total_amount: Total amount of high-risk transactions
                - transactions: List of high-risk transaction details
        """
        high_risk = data[data['fraud_probability'] >= self.threshold]
        return {
            'num_transactions': len(high_risk),
            'total_amount': high_risk['amount'].sum(),
            'transactions': high_risk[['payment_id', 'amount', 'fraud_probability']].to_dict('records')
        }

class FraudMonitoringSystem:
    """
    Fraud monitoring system with real-time pattern detection and alerting.
    
    This class implements a comprehensive fraud monitoring solution with:
    - Real-time transaction monitoring
    - Pattern detection and anomaly identification
    - Performance metric tracking
    - Automated alerting
    - Visualization of trends
    - Reporting capabilities
    
    Attributes:
        config (MonitoringConfig): System configuration
        metrics_history (List[MetricsDict]): Historical performance metrics
        alert_strategies (List[AlertStrategy]): Active alert strategies
        logger (logging.Logger): System logger
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the fraud monitoring system.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration and initialize components
        self.config = self._load_config(config_path)
        self.metrics_history: List[MetricsDict] = []
        self.alert_strategies: List[AlertStrategy] = [
            HighRiskTransactionAlert(self.config.alert_threshold)
        ]
        
        # Setup logging and directories
        self._setup_logging()
        self._setup_output_directories()
        
        self.logger.info("Fraud Monitoring System initialized successfully")

    def _load_config(self, config_path: Union[str, Path]) -> MonitoringConfig:
        """
        Load and validate monitoring system configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            MonitoringConfig: Validated configuration object
            
        Raises:
            ValueError: If required configuration parameters are missing
            yaml.YAMLError: If YAML file is malformed
        """
        try:
            # Load and parse YAML file
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = [
                'alert_threshold',
                'monitoring_window',
                'email_recipients',
                'alert_rules',
                'visualization_settings'
            ]
            
            missing_sections = [
                section for section in required_sections 
                if section not in config_dict
            ]
            
            if missing_sections:
                raise ValueError(
                    f"Missing required configuration sections: {missing_sections}"
                )
            
            return MonitoringConfig(**config_dict)
            
        except Exception as e:
            raise ValueError(f"Error loading configuration: {str(e)}")

    def _setup_logging(self) -> None:
        """
        Configure logging with rotation and formatting.
        
        Creates a daily rotating log file with timestamp and level information.
        """
        log_filename = f'logs/fraud_monitoring_{datetime.now().strftime("%Y%m%d")}.log'
        Path(log_filename).parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        self.logger = logging.getLogger('FraudMonitor')

    def _setup_output_directories(self) -> None:
        """
        Create necessary output directories for storing results.
        
        Creates directories for:
        - outputs: General output files
        - figures: Visualization plots
        - reports: Daily reports
        - alerts: Alert records
        """
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
        Calculate comprehensive performance metrics for the fraud detection system.
        
        Calculates various metrics including:
        - Accuracy, Precision, Recall, F1-score
        - ROC AUC and PR AUC
        - True/False Positive/Negative counts
        
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
            # Validate input arrays
            if len(y_true) != len(y_pred) or len(y_true) != len(y_prob):
                raise ValueError("Input arrays must have the same length")

            # Calculate confusion matrix metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate derived metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate ROC and PR curves
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            
            # Compile metrics
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
            
            # Store metrics history
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
        
        Analyzes various patterns including:
        - Transaction velocity
        - Amount patterns
        - Risk distribution
        - Temporal patterns
        
        Args:
            transactions: DataFrame of recent transactions
            window_size: Time window for pattern analysis (default from config)
            
        Returns:
            Dictionary containing detected patterns and anomalies
        """
        try:
            # Set analysis window
            window = window_size or self.config.monitoring_window
            transactions = transactions.copy()
            
            # Prepare time index
            transactions['timestamp'] = pd.to_datetime(transactions['adjusted_pmt_created_at'])
            transactions.set_index('timestamp', inplace=True)
            
            # Analyze patterns
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
        """
        Calculate transaction velocity patterns for different entities.
        
        Args:
            transactions: DataFrame of transactions
            
        Returns:
            Dictionary containing velocity statistics by IP and consumer
        """
        return {
            'per_ip': transactions.groupby('hashed_ip').size().describe().to_dict(),
            'per_consumer': transactions.groupby('hashed_consumer_id').size().describe().to_dict(),
            'hourly_volume': transactions.resample('H').size().describe().to_dict()
        }

    def _analyze_amount_patterns(self, transactions: TransactionData) -> Dict[str, Any]:
        """
        Analyze transaction amount patterns and detect anomalies.
        
        Args:
            transactions: DataFrame of transactions
            
        Returns:
            Dictionary containing amount statistics and anomalies
        """
        return {
            'statistics': transactions['amount'].describe().to_dict(),
            'anomalies': self._detect_amount_anomalies(transactions)
        }

    def run_monitoring_cycle(
        self,
        transactions: TransactionData,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> None:
        """
        Run complete monitoring cycle including metrics, patterns, and alerts.
        
        This is the main monitoring function that:
        1. Calculates performance metrics
        2. Monitors patterns
        3. Generates reports
        4. Checks for alerts
        5. Creates visualizations
        
        Args:
            transactions: Current transaction data
            y_true: True fraud labels
            y_pred: Predicted fraud labels
            y_prob: Fraud probabilities
            
        Raises:
            Exception: If any step in the monitoring cycle fails
        """
        try:
            # Step 1: Calculate performance metrics
            metrics = self.calculate_performance_metrics(y_true, y_pred, y_prob)
            
            # Step 2: Monitor patterns
            patterns = self.monitor_real_time_patterns(transactions)
            
            # Step 3: Generate report
            report = self.generate_daily_report(transactions, metrics)
            
            # Step 4: Check for alerts
            self.check_alerts(transactions)
            
            # Step 5: Generate visualizations
            self.visualize_trends(
                transactions,
                f'figures/trends_{datetime.now().strftime("%Y%m%d")}.png'
            )
            
            # Save cycle results
            self._save_cycle_results(metrics, patterns)
            
            self.logger.info("Monitoring cycle completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {str(e)}")
            raise

def main():
    """
    Main execution function for the fraud monitoring system.
    
    This function:
    1. Initializes the monitoring system
    2. Loads transaction data
    3. Runs the monitoring cycle
    4. Handles any errors during execution
    
    Environment Variables:
        FRAUD_DETECTION_ENV: Environment setting ('prod', 'dev', 'test')
        
    Configuration:
        Uses config/monitoring_config_{env}.yml based on environment
    """
    try:
        # Get environment setting
        env = os.getenv('FRAUD_DETECTION_ENV', 'dev')
        
        # Determine config file
        config_files = {
            'prod': 'monitoring_config_prod.yml',
            'dev': 'monitoring_config_dev.yml',
            'test': 'monitoring_config_test.yml'
        }
        config_file = config_files.get(env, 'monitoring_config.yml')
        config_path = os.path.join('config', config_file)
        
        # Initialize monitoring system
        print(f"Initializing Fraud Monitoring System with config: {config_file}")
        monitor = FraudMonitoringSystem(config_path)
        
        # Load transactions
        print("Loading transaction data...")
        transactions = pd.read_csv('data_scientist_fraud_20241009.csv')
        
        # Prepare monitoring data
        print("Preparing monitoring data...")
        y_true = transactions['fraud_flag'].fillna(0)
        y_prob = transactions['fraud_probability']
        y_pred = (y_prob >= monitor.config.alert_threshold).astype(int)
        
        # Run monitoring cycle
        print("Running monitoring cycle...")
        monitor.run_monitoring_cycle(
            transactions=transactions,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob
        )
        
        print("\nMonitoring cycle completed successfully!")
        print(f"Results saved in: {os.path.abspath('outputs')}")
        print(f"Reports available in: {os.path.abspath('reports')}")
        
    except FileNotFoundError as e:
        logging.error(f"Configuration or data file not found: {str(e)}")
        print(f"Error: Required file not found - {str(e)}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Invalid data or configuration: {str(e)}")
        print(f"Error: Invalid data or configuration - {str(e)}")
        sys.exit(2)
    except Exception as e:
        logging.error(f"Unexpected error in main execution: {str(e)}")
        print(f"Error: Unexpected error occurred - {str(e)}")
        sys.exit(3)

if __name__ == "__main__":
    # Set up console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    # Run main function
    main()
