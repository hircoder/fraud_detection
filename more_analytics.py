# more analytics

class AdvancedAnalysis:
    def __init__(self, df):
        self.df = df
        self.merchant_insights = None
        self.temporal_insights = None
        self.statistical_patterns = None

    def analyze_merchant_patterns(self):
        """
        Detailed merchant-specific analysis
        """
        merchant_stats = {}
        for merchant in self.df['merchant_name'].unique():
            merchant_data = self.df[self.df['merchant_name'] == merchant]
            
            # Basic statistics
            stats = {
                'total_transactions': len(merchant_data),
                'fraud_rate': merchant_data['fraud_flag'].mean() * 100,
                'avg_transaction': merchant_data['amount'].mean(),
                'total_volume': merchant_data['amount'].sum(),
                
                # Risk metrics
                'new_account_rate': (merchant_data['merchant_account_age'] == 0).mean(),
                'high_risk_rate': (merchant_data['account_age_risk'] > 0.5).mean(),
                
                # Temporal patterns
                'peak_hours': merchant_data.groupby('payment_hour')['fraud_flag'].mean().nlargest(3).index.tolist(),
                'weekend_ratio': merchant_data[merchant_data['is_weekend'] == 1]['fraud_flag'].mean() / 
                                merchant_data[merchant_data['is_weekend'] == 0]['fraud_flag'].mean(),
                
                # User demographics
                'age_distribution': merchant_data['consumer_age'].describe().to_dict(),
                'gender_distribution': merchant_data['consumer_gender'].value_counts(normalize=True).to_dict()
            }
            
            # Velocity metrics
            stats.update({
                'avg_tx_per_user': merchant_data.groupby('hashed_consumer_id').size().mean(),
                'high_velocity_rate': (merchant_data['tx_count_1H'] > 2).mean(),
                'repeat_customer_rate': len(merchant_data[merchant_data['merchant_account_age'] > 30]) / len(merchant_data)
            })
            
            merchant_stats[merchant] = stats
            
        self.merchant_insights = merchant_stats
        return merchant_stats

    def analyze_temporal_patterns(self):
        """
        Comprehensive temporal analysis
        """
        temporal_insights = {}
        
        # Hourly patterns
        hourly = self.df.groupby('payment_hour').agg({
            'fraud_flag': ['count', 'mean', 'sum'],
            'amount': ['mean', 'sum']
        }).round(4)
        
        # Daily patterns
        daily = self.df.groupby('payment_day').agg({
            'fraud_flag': ['count', 'mean', 'sum'],
            'amount': ['mean', 'sum']
        }).round(4)
        
        # Time-based risk scores
        time_risk = pd.DataFrame()
        time_risk['hour_risk'] = self.df.groupby('payment_hour')['fraud_flag'].mean()
        time_risk['day_risk'] = self.df.groupby('payment_day')['fraud_flag'].mean()
        time_risk['weekend_risk'] = self.df.groupby('is_weekend')['fraud_flag'].mean()
        
        # Velocity analysis
        velocity_stats = {
            'high_velocity_fraud_rate': self.df[self.df['tx_count_1H'] > 2]['fraud_flag'].mean(),
            'normal_velocity_fraud_rate': self.df[self.df['tx_count_1H'] <= 2]['fraud_flag'].mean(),
            'velocity_risk_ratio': (
                self.df[self.df['tx_count_1H'] > 2]['fraud_flag'].mean() /
                self.df[self.df['tx_count_1H'] <= 2]['fraud_flag'].mean()
            )
        }
        
        temporal_insights = {
            'hourly_patterns': hourly.to_dict(),
            'daily_patterns': daily.to_dict(),
            'time_risk_scores': time_risk.to_dict(),
            'velocity_statistics': velocity_stats
        }
        
        self.temporal_insights = temporal_insights
        return temporal_insights

    def analyze_statistical_patterns(self):
        """
        Statistical analysis of fraud patterns
        """
        patterns = {}
        
        # Amount distribution analysis
        amount_stats = {
            'fraud_amount_stats': self.df[self.df['fraud_flag'] == 1]['amount'].describe().to_dict(),
            'normal_amount_stats': self.df[self.df['fraud_flag'] == 0]['amount'].describe().to_dict(),
            'amount_percentiles': {
                f"p{i}": np.percentile(self.df['amount'], i)
                for i in [25, 50, 75, 90, 95, 99]
            }
        }
        
        # Risk factor analysis
        risk_correlations = {}
        risk_features = ['account_age_risk', 'amount_risk', 'consumer_age_risk']
        for feature in risk_features:
            risk_correlations[feature] = {
                'fraud_correlation': self.df[feature].corr(self.df['fraud_flag']),
                'risk_distribution': self.df.groupby(feature)['fraud_flag'].mean().to_dict()
            }
        
        # User behavior patterns
        user_patterns = {
            'new_user_fraud_rate': self.df[self.df['account_age_hours'] < 24]['fraud_flag'].mean(),
            'established_user_fraud_rate': self.df[self.df['account_age_hours'] >= 24]['fraud_flag'].mean(),
            'age_group_risks': self.df.groupby(pd.qcut(self.df['consumer_age'], 5))['fraud_flag'].mean().to_dict()
        }
        
        patterns = {
            'amount_analysis': amount_stats,
            'risk_correlations': risk_correlations,
            'user_patterns': user_patterns
        }
        
        self.statistical_patterns = patterns
        return patterns
