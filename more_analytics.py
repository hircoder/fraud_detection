# more_analytics.py
"""
Advanced Analytics Module for Fraud Detection System

This module provides comprehensive analytics capabilities for analyzing fraud patterns,
merchant behavior, temporal trends, and statistical distributions in transaction data.

Key Features:
    - Merchant-specific pattern analysis
    - Temporal pattern detection
    - Statistical pattern analysis
    - Risk correlation analysis
    - User behavior tracking

Usage:
    analyzer = AdvancedAnalysis(transaction_dataframe)
    
    # Get merchant insights
    merchant_patterns = analyzer.analyze_merchant_patterns()
    
    # Get temporal insights
    temporal_patterns = analyzer.analyze_temporal_patterns()
    
    # Get statistical patterns
    statistical_patterns = analyzer.analyze_statistical_patterns()

Author: Hiroshi
Date: November 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

class AdvancedAnalysis:
    """
    Advanced Analytics class for fraud detection pattern analysis.
    
    This class provides three main types of analysis:
    1. Merchant Pattern Analysis: Analyzes merchant-specific fraud patterns and risk indicators
    2. Temporal Pattern Analysis: Analyzes time-based patterns in fraudulent activities
    3. Statistical Pattern Analysis: Provides statistical insights into fraud patterns
    
    Attributes:
        df (pd.DataFrame): Input transaction data
        merchant_insights (Dict): Stored merchant analysis results
        temporal_insights (Dict): Stored temporal analysis results
        statistical_patterns (Dict): Stored statistical analysis results
        
    Required DataFrame Columns:
        - merchant_name: Name of merchant
        - fraud_flag: Binary indicator of fraud (1 = fraud, 0 = normal)
        - amount: Transaction amount
        - merchant_account_age: Age of merchant account
        - account_age_risk: Risk score based on account age
        - payment_hour: Hour of transaction
        - is_weekend: Weekend indicator
        - consumer_age: Age of consumer
        - consumer_gender: Gender of consumer
        - hashed_consumer_id: Unique consumer identifier
        - tx_count_1H: Transaction count in past hour
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Advanced Analysis system.
        
        Args:
            df (pd.DataFrame): Transaction data for analysis
            
        Raises:
            ValueError: If required columns are missing from DataFrame
        """
        self._validate_dataframe(df)
        self.df = df
        self.merchant_insights = None
        self.temporal_insights = None
        self.statistical_patterns = None

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate that the input DataFrame has all required columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = [
            'merchant_name', 'fraud_flag', 'amount', 'merchant_account_age',
            'account_age_risk', 'payment_hour', 'is_weekend', 'consumer_age',
            'consumer_gender', 'hashed_consumer_id', 'tx_count_1H'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def analyze_merchant_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform detailed merchant-specific analysis of fraud patterns.
        
        This method analyzes each merchant's:
        - Basic transaction statistics
        - Risk metrics
        - Temporal patterns
        - User demographics
        - Velocity metrics
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary containing merchant-specific analytics
            {
                'merchant_name': {
                    'total_transactions': int,
                    'fraud_rate': float,
                    'avg_transaction': float,
                    'total_volume': float,
                    'new_account_rate': float,
                    'high_risk_rate': float,
                    'peak_hours': List[int],
                    'weekend_ratio': float,
                    'age_distribution': Dict,
                    'gender_distribution': Dict,
                    'avg_tx_per_user': float,
                    'high_velocity_rate': float,
                    'repeat_customer_rate': float
                }
            }
        """
        merchant_stats = {}
        
        for merchant in self.df['merchant_name'].unique():
            # Filter data for current merchant
            merchant_data = self.df[self.df['merchant_name'] == merchant]
            
            # Calculate basic statistics
            stats = {
                'total_transactions': len(merchant_data),
                'fraud_rate': merchant_data['fraud_flag'].mean() * 100,
                'avg_transaction': merchant_data['amount'].mean(),
                'total_volume': merchant_data['amount'].sum(),
            }
            
            # Calculate risk metrics
            stats.update({
                'new_account_rate': (merchant_data['merchant_account_age'] == 0).mean(),
                'high_risk_rate': (merchant_data['account_age_risk'] > 0.5).mean(),
            })
            
            # Analyze temporal patterns
            stats.update({
                'peak_hours': merchant_data.groupby('payment_hour')['fraud_flag']
                    .mean().nlargest(3).index.tolist(),
                'weekend_ratio': (
                    merchant_data[merchant_data['is_weekend'] == 1]['fraud_flag'].mean() /
                    merchant_data[merchant_data['is_weekend'] == 0]['fraud_flag'].mean()
                ),
            })
            
            # Analyze user demographics
            stats.update({
                'age_distribution': merchant_data['consumer_age'].describe().to_dict(),
                'gender_distribution': merchant_data['consumer_gender']
                    .value_counts(normalize=True).to_dict(),
            })
            
            # Calculate velocity metrics
            stats.update({
                'avg_tx_per_user': merchant_data.groupby('hashed_consumer_id').size().mean(),
                'high_velocity_rate': (merchant_data['tx_count_1H'] > 2).mean(),
                'repeat_customer_rate': len(merchant_data[merchant_data['merchant_account_age'] > 30]) / len(merchant_data)
            })
            
            merchant_stats[merchant] = stats
            
        self.merchant_insights = merchant_stats
        return merchant_stats

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """
        Perform comprehensive temporal analysis of fraud patterns.
        
        Analyzes patterns across different time periods:
        - Hourly patterns
        - Daily patterns
        - Time-based risk scores
        - Velocity statistics
        
        Returns:
            Dict[str, Any]: Dictionary containing temporal insights
            {
                'hourly_patterns': Dict,  # Hourly aggregated metrics
                'daily_patterns': Dict,   # Daily aggregated metrics
                'time_risk_scores': {     # Risk scores by time period
                    'hour_risk': Series,
                    'day_risk': Series,
                    'weekend_risk': Series
                },
                'velocity_statistics': {   # Transaction velocity metrics
                    'high_velocity_fraud_rate': float,
                    'normal_velocity_fraud_rate': float,
                    'velocity_risk_ratio': float
                }
            }
        """
        # Calculate hourly patterns
        hourly = self.df.groupby('payment_hour').agg({
            'fraud_flag': ['count', 'mean', 'sum'],
            'amount': ['mean', 'sum']
        }).round(4)
        
        # Calculate daily patterns
        daily = self.df.groupby('payment_day').agg({
            'fraud_flag': ['count', 'mean', 'sum'],
            'amount': ['mean', 'sum']
        }).round(4)
        
        # Calculate time-based risk scores
        time_risk = pd.DataFrame()
        time_risk['hour_risk'] = self.df.groupby('payment_hour')['fraud_flag'].mean()
        time_risk['day_risk'] = self.df.groupby('payment_day')['fraud_flag'].mean()
        time_risk['weekend_risk'] = self.df.groupby('is_weekend')['fraud_flag'].mean()
        
        # Analyze transaction velocity patterns
        velocity_stats = {
            'high_velocity_fraud_rate': self.df[self.df['tx_count_1H'] > 2]['fraud_flag'].mean(),
            'normal_velocity_fraud_rate': self.df[self.df['tx_count_1H'] <= 2]['fraud_flag'].mean(),
            'velocity_risk_ratio': (
                self.df[self.df['tx_count_1H'] > 2]['fraud_flag'].mean() /
                self.df[self.df['tx_count_1H'] <= 2]['fraud_flag'].mean()
            )
        }
        
        # Combine all temporal insights
        temporal_insights = {
            'hourly_patterns': hourly.to_dict(),
            'daily_patterns': daily.to_dict(),
            'time_risk_scores': time_risk.to_dict(),
            'velocity_statistics': velocity_stats
        }
        
        self.temporal_insights = temporal_insights
        return temporal_insights

    def analyze_statistical_patterns(self) -> Dict[str, Any]:
        """
        Perform statistical analysis of fraud patterns.
        
        Analyzes:
        - Transaction amount distributions
        - Risk factor correlations
        - User behavior patterns
        
        Returns:
            Dict[str, Any]: Dictionary containing statistical patterns
            {
                'amount_analysis': {
                    'fraud_amount_stats': Dict,    # Amount statistics for fraudulent transactions
                    'normal_amount_stats': Dict,   # Amount statistics for normal transactions
                    'amount_percentiles': Dict     # Amount distribution percentiles
                },
                'risk_correlations': {            # Correlation analysis for risk factors
                    'feature_name': {
                        'fraud_correlation': float,
                        'risk_distribution': Dict
                    }
                },
                'user_patterns': {                # User behavior analysis
                    'new_user_fraud_rate': float,
                    'established_user_fraud_rate': float,
                    'age_group_risks': Dict
                }
            }
        """
        # Analyze amount distributions
        amount_stats = {
            'fraud_amount_stats': self.df[self.df['fraud_flag'] == 1]['amount'].describe().to_dict(),
            'normal_amount_stats': self.df[self.df['fraud_flag'] == 0]['amount'].describe().to_dict(),
            'amount_percentiles': {
                f"p{i}": np.percentile(self.df['amount'], i)
                for i in [25, 50, 75, 90, 95, 99]
            }
        }
        
        # Analyze risk factor correlations
        risk_correlations = {}
        risk_features = ['account_age_risk', 'amount_risk', 'consumer_age_risk']
        for feature in risk_features:
            risk_correlations[feature] = {
                'fraud_correlation': self.df[feature].corr(self.df['fraud_flag']),
                'risk_distribution': self.df.groupby(feature)['fraud_flag'].mean().to_dict()
            }
        
        # Analyze user behavior patterns
        user_patterns = {
            'new_user_fraud_rate': self.df[self.df['account_age_hours'] < 24]['fraud_flag'].mean(),
            'established_user_fraud_rate': self.df[self.df['account_age_hours'] >= 24]['fraud_flag'].mean(),
            'age_group_risks': self.df.groupby(pd.qcut(self.df['consumer_age'], 5))['fraud_flag'].mean().to_dict()
        }
        
        # Combine all statistical patterns
        patterns = {
            'amount_analysis': amount_stats,
            'risk_correlations': risk_correlations,
            'user_patterns': user_patterns
        }
        
        self.statistical_patterns = patterns
        return patterns
