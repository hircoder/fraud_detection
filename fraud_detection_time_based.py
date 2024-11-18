import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    fbeta_score,
    confusion_matrix
)
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

import logging
import joblib

# Configure logging
logging.basicConfig(filename='fraud_detection.log', level=logging.INFO)


class FraudDetectionSystem:
    def __init__(self):
        """
        Initialize the FraudDetectionSystem with default values.
        """
        self.model = None
        self.preprocessor = None
        self.features = None
        self.optimal_threshold = 0.5  # Default threshold

    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the fraud detection dataset.

        Parameters:
        - file_path (str): The path to the CSV data file.

        Returns:
        - df (DataFrame): The preprocessed DataFrame.
        """
        try:
            # Load the data
            df = pd.read_csv(file_path)

            # Convert timestamps to datetime
            df['adjusted_pmt_created_at'] = pd.to_datetime(df['adjusted_pmt_created_at'])
            df['adjusted_acc_created_at'] = pd.to_datetime(df['adjusted_acc_created_at'])

            # Fill missing values in features
            df.fillna({
                'device': 'Unknown',
                'version': 'Unknown',
                'consumer_gender': 'Unknown',
                'consumer_age': df['consumer_age'].median(),
                'consumer_phone_age': df['consumer_phone_age'].median(),
                'merchant_account_age': df['merchant_account_age'].median(),
                'ltv': df['ltv'].median(),
            }, inplace=True)

            # Handle fraud_flag missing values
            # Assuming nulls are non-fraudulent transactions
            df['fraud_flag'] = df['fraud_flag'].fillna(0).astype(int)

            # Feature Engineering
            df = self.create_features(df)

            logging.info("Data loaded and preprocessed successfully.")
            return df

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def create_features(self, df):
        """
        Create new features for fraud detection.
        """
        # Sort data by consumer ID and timestamp
        df = df.sort_values(['hashed_consumer_id', 'adjusted_pmt_created_at'])

        # Time-based features
        df['account_age_hours'] = (
            df['adjusted_pmt_created_at'] - df['adjusted_acc_created_at']
        ).dt.total_seconds() / 3600
        df['payment_hour'] = df['adjusted_pmt_created_at'].dt.hour
        df['payment_day'] = df['adjusted_pmt_created_at'].dt.day
        df['is_weekend'] = df['adjusted_pmt_created_at'].dt.weekday.isin([5, 6]).astype(int)

        # Transaction velocity with time windows
        # Set the datetime index
        df.set_index('adjusted_pmt_created_at', inplace=True)

        # Calculate transaction counts in the past 1 hour and 24 hours, excluding the current transaction
        df['tx_count_1H'] = df.groupby('hashed_consumer_id')['payment_id'].apply(
            lambda x: x.shift().rolling('1H').count()
        ).reset_index(level=0, drop=True)

        df['tx_count_24H'] = df.groupby('hashed_consumer_id')['payment_id'].apply(
            lambda x: x.shift().rolling('24H').count()
        ).reset_index(level=0, drop=True)

        # Reset index
        df.reset_index(inplace=True)

        # Fill any NaN values resulting from rolling computations
        df['tx_count_1H'] = df['tx_count_1H'].fillna(0)
        df['tx_count_24H'] = df['tx_count_24H'].fillna(0)

        # Amount features with normalization
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        df['amount_percentile'] = df['amount'].rank(pct=True)

        # Rolling mean features to prevent data leakage
        df['amount_rolling_mean'] = df.groupby('hashed_consumer_id')['amount'].apply(
            lambda x: x.shift().rolling(window=3, min_periods=1).mean()
        )
        df['account_age_hours_rolling_mean'] = df.groupby('hashed_consumer_id')['account_age_hours'].apply(
            lambda x: x.shift().rolling(window=3, min_periods=1).mean()
        )

        # Fill NaN values in rolling means
        df['amount_rolling_mean'] = df['amount_rolling_mean'].fillna(df['amount'].mean())
        df['account_age_hours_rolling_mean'] = df['account_age_hours_rolling_mean'].fillna(df['account_age_hours'].mean())

        # Identity consistency features
        df['email_match'] = (df['hashed_buyer_email'] == df['hashed_consumer_email']).astype(int)
        df['phone_match'] = (df['hashed_buyer_phone'] == df['hashed_consumer_phone']).astype(int)

        # Risk scoring features
        df['account_age_risk'] = np.where(df['account_age_hours'] < 1, 1,
                                          np.where(df['account_age_hours'] < 24, 0.5, 0))
        df['amount_risk'] = np.where(df['amount'] > 10000, 1,
                                     np.where(df['amount'] > 5000, 0.5, 0))

        # Consumer age risk
        df['consumer_age_risk'] = np.where(df['consumer_age'] < 25, 1,
                                           np.where(df['consumer_age'] > 60, 1, 0))

        logging.info("Feature engineering completed.")
        return df

    def prepare_features(self, df):
        """
        Prepare features for modeling.
        """
        try:
            # Select features for modeling
            feature_columns = [
                'account_age_hours', 'payment_hour', 'payment_day', 'is_weekend',
                'tx_count_1H', 'tx_count_24H',
                'amount_zscore', 'amount_percentile',
                'amount_rolling_mean', 'account_age_hours_rolling_mean',
                'email_match', 'phone_match', 'account_age_risk', 'amount_risk',
                'consumer_phone_age', 'merchant_account_age', 'ltv', 'consumer_age',
                'consumer_age_risk',
                'device', 'version', 'merchant_name', 'consumer_gender'
            ]

            # Store feature names for later use
            self.features = feature_columns.copy()

            # Create copy of selected features
            X = df[feature_columns].copy()

            # Define numerical columns
            numerical_columns = [
                'account_age_hours', 'payment_hour', 'payment_day', 'is_weekend',
                'tx_count_1H', 'tx_count_24H',
                'amount_zscore', 'amount_percentile',
                'amount_rolling_mean', 'account_age_hours_rolling_mean',
                'email_match', 'phone_match', 'account_age_risk', 'amount_risk',
                'consumer_phone_age', 'merchant_account_age', 'ltv', 'consumer_age',
                'consumer_age_risk'
            ]

            # Define categorical columns
            categorical_columns = ['device', 'version', 'merchant_name', 'consumer_gender']

            # Handle numerical missing values
            for col in numerical_columns:
                if col in X.columns:
                    median_value = X[col].median()
                    X[col] = X[col].fillna(median_value)
                    X[col] = X[col].astype(float)

            # Handle categorical missing values
            for col in categorical_columns:
                X[col] = X[col].fillna('Unknown')
                X[col] = X[col].astype(str)

            # Check for any remaining NaN values
            if X.isnull().any().any():
                null_counts = X.isnull().sum()
                logging.warning(f"Found remaining null values:\n{null_counts[null_counts > 0]}")
                raise ValueError("Found unexpected null values in features")

            logging.info(f"Feature preparation completed. Shape: {X.shape}")
            return X, numerical_columns, categorical_columns

        except Exception as e:
            logging.error(f"Error in prepare_features: {str(e)}")
            raise

    def train_model(self, X_data, y):
        """
        Train the fraud detection model and save visualizations.
        """
        try:
            # Create output directories
            os.makedirs('figures', exist_ok=True)
            os.makedirs('outputs', exist_ok=True)

            # Unpack prepared data
            X, numerical_columns, categorical_columns = X_data

            # Define preprocessing pipelines
            numerical_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_columns),
                    ('cat', categorical_transformer, categorical_columns)
                ]
            )

            # Preprocess the data
            X_preprocessed = preprocessor.fit_transform(X)

            # Store the preprocessor for use in prediction
            self.preprocessor = preprocessor

            # Handle class imbalance using scale_pos_weight
            if sum(y == 1) == 0:
                scale_pos_weight = 1
            else:
                scale_pos_weight = sum(y == 0) / sum(y == 1)

            # Train the final model
            self.model = xgb.XGBClassifier(
                learning_rate=0.01,
                n_estimators=100,
                max_depth=4,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                gamma=1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )

            self.model.fit(X_preprocessed, y)

            logging.info("Model training completed.")

            return self.model

        except Exception as e:
            logging.error(f"Error in train_model: {str(e)}")
            raise

    def analyze_feature_correlations(self, X, y):
        """Analyze and plot feature correlations with the target."""
        # Combine X and y
        df = X.copy()
        df['fraud_flag'] = y.values

        # Calculate correlation matrix
        corr_matrix = df.corr()
        target_corr = corr_matrix['fraud_flag'].drop('fraud_flag')

        # Plot correlations
        plt.figure(figsize=(10, 8))
        target_corr.sort_values(ascending=False).plot(kind='bar')
        plt.title('Feature Correlations with Target')
        plt.tight_layout()
        plt.savefig('figures/feature_correlations.png')
        plt.close()

        return target_corr

    def predict_fraud_probability(self, X_data):
        """
        Predict fraud probability for new transactions.

        Parameters:
        - X_data (tuple): The feature matrix and related info.

        Returns:
        - probabilities (ndarray): The array of fraud probabilities.
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model or preprocessor not trained yet!")

        # Unpack prepared data
        X, _, _ = X_data

        # Preprocess the data
        X_preprocessed = self.preprocessor.transform(X)

        # Predict probabilities
        probabilities = self.model.predict_proba(X_preprocessed)[:, 1]
        logging.info("Fraud probabilities predicted.")
        return probabilities

    def save_model(self, filepath):
        """
        Save the trained model and preprocessor to a file.
        """
        joblib.dump({'model': self.model, 'preprocessor': self.preprocessor}, filepath)
        logging.info(f"Model and preprocessor saved to {filepath}")

    def calculate_business_impact(self, df, threshold):
        """
        Calculate the business impact of the fraud detection system.

        Parameters:
        - df (DataFrame): The DataFrame containing transactions and predictions.
        - threshold (float): The threshold for classifying transactions as fraud.

        Returns:
        - impact (dict): A dictionary containing business impact metrics.
        """
        # Placeholder implementation
        return {}

    def _save_model_plots(self, y_test, y_pred_optimal, y_pred_proba, metrics):
        """
        Save ROC curve, Precision-Recall curve, and confusion matrix plots.

        Parameters:
        - y_test (Series): True labels.
        - y_pred_optimal (ndarray): Predicted labels using the optimal threshold.
        - y_pred_proba (ndarray): Predicted probabilities.
        - metrics (dict): Dictionary of performance metrics.
        """
        # Save ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["auc_roc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('figures/01_roc_curve.png')
        plt.close()

        # Save Precision-Recall curve
        plt.figure(figsize=(8, 6))
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall_vals, precision_vals, label=f'AP = {metrics["auc_pr"]:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig('figures/02_precision_recall_curve.png')
        plt.close()

        # Save confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred_optimal)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('figures/03_confusion_matrix.png')
        plt.close()


def main():
    """
    Main execution function
    """
    # Initialize the fraud detection system
    fraud_system = FraudDetectionSystem()

    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = fraud_system.load_and_preprocess_data('data_scientist_fraud_20241009.csv')

        # Handle fraud_flag missing values
        df['fraud_flag'] = df['fraud_flag'].fillna(0).astype(int)

        # Convert adjusted_pmt_created_at to date
        df['payment_date'] = df['adjusted_pmt_created_at'].dt.date

        # Analyze fraud cases over time
        fraud_counts = df.groupby('payment_date')['fraud_flag'].sum()
        print("\nFraud cases per date:")
        print(fraud_counts[fraud_counts > 0])

        # Set the cutoff date to ensure both classes are in the test set
        cutoff_date = '2021-04-27'

        # Split data into training and test sets based on time
        train_df = df[df['adjusted_pmt_created_at'] < cutoff_date]
        test_df = df[df['adjusted_pmt_created_at'] >= cutoff_date]

        # Verify class distribution
        print("\nTraining set class distribution:")
        print(train_df['fraud_flag'].value_counts())
        print("\nTest set class distribution:")
        print(test_df['fraud_flag'].value_counts())

        # Proceed only if both sets contain both classes
        if train_df['fraud_flag'].nunique() < 2 or test_df['fraud_flag'].nunique() < 2:
            raise ValueError("Not enough classes in training or test set. Adjust the cutoff date or use a different method.")

        # Prepare features and target for training
        print("\nPreparing training features...")
        X_train_data = fraud_system.prepare_features(train_df)
        X_train, _, _ = X_train_data
        y_train = train_df['fraud_flag']

        # Prepare features and target for testing
        print("Preparing test features...")
        X_test_data = fraud_system.prepare_features(test_df)
        X_test, _, _ = X_test_data
        y_test = test_df['fraud_flag']

        # Analyze feature correlations on training data
        feature_correlations = fraud_system.analyze_feature_correlations(X_train, y_train)
        print("\nTop feature correlations with target:")
        print(feature_correlations.sort_values(ascending=False).head(10))

        # Train model
        print("\nTraining model...")
        model = fraud_system.train_model(X_train_data, y_train)

        # Predict fraud probabilities on test data
        print("\nMaking predictions on test data...")
        test_df['fraud_probability'] = fraud_system.predict_fraud_probability(X_test_data)

        # Evaluate model on test data
        y_pred_proba = test_df['fraud_probability']
        y_pred = (y_pred_proba >= fraud_system.optimal_threshold).astype(int)

        # Find optimal threshold on validation data (if available)
        # Since we don't have validation data, we'll use default threshold

        # Calculate and print metrics
        if y_test.nunique() < 2:
            print("\nWarning: Only one class present in y_test. Some metrics may not be defined.")
            auc_roc = None
        else:
            auc_roc = roc_auc_score(y_test, y_pred_proba)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'f2': fbeta_score(y_test, y_pred, beta=2),
            'auc_roc': auc_roc,
            'auc_pr': average_precision_score(y_test, y_pred_proba),
            'optimal_threshold': fraud_system.optimal_threshold
        }

        print("\nModel Performance Metrics on Test Data:")
        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: Undefined (only one class present in y_test)")

        # Save metrics
        with open('outputs/model_metrics.txt', 'w') as f:
            f.write("Model Performance Metrics on Test Data:\n")
            for metric, value in metrics.items():
                if value is not None:
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: Undefined (only one class present in y_test)\n")

        # Save high-risk transactions
        high_risk_transactions = test_df[
            test_df['fraud_probability'] >= fraud_system.optimal_threshold
        ][['payment_id', 'fraud_probability', 'amount', 'hashed_consumer_id']]

        high_risk_transactions.to_csv('outputs/high_risk_transactions.csv', index=False)
        print(f"\nIdentified {len(high_risk_transactions)} high-risk transactions")

        # Calculate and save impact analysis
        impact = fraud_system.calculate_business_impact(test_df, threshold=fraud_system.optimal_threshold)
        with open('outputs/business_impact.txt', 'w') as f:
            f.write("Business Impact Analysis:\n")
            for key, value in impact.items():
                f.write(f"{key}: {value}\n")

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total transactions: {len(df)}")
        print(f"Total fraudulent transactions: {df['fraud_flag'].sum()}")
        print(f"Fraud rate: {df['fraud_flag'].mean()*100:.4f}%")
        print(f"Optimal threshold: {fraud_system.optimal_threshold:.4f}")

        # Save model and components
        print("\nSaving model and components...")
        fraud_system.save_model('outputs/fraud_detection_model.pkl')

        print("\nAnalysis completed successfully!")

    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        logging.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
