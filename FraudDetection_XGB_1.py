#FraudDetection_XGB_1.py

# Part 1: Imports and Setup
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

import sklearn
sklearn.set_config(assume_finite=True)
from joblib import parallel_backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, average_precision_score, fbeta_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek  
from sklearn.metrics import make_scorer 
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import logging
import joblib
import gc
from datetime import datetime
import traceback

# Configure logging and warnings
warnings.filterwarnings('ignore')
logging.basicConfig(filename='fraud_detection_XGB_1.log', level=logging.INFO)

# Part 2: Feature Engineering Class
class FeatureEngineering:
    def __init__(self):
        self.feature_weights = {
            'account_age': 0.497,
            'device': 0.250,
            'velocity': 0.150,
            'identity': 0.053,
            'amount': 0.040
        }

    def validate_timestamps(self, df):
        """Validate and convert timestamp columns"""
        timestamp_columns = ['adjusted_pmt_created_at', 'adjusted_acc_created_at']
        
        for col in timestamp_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required timestamp column: {col}")
                
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception as e:
                    raise ValueError(f"Failed to convert {col} to datetime: {str(e)}")
                    
        return df

    def _calculate_rolling_count(self, df, column, window):
        """Enhanced rolling count calculator with index preservation"""
        try:
            # Create copy to preserve original data
            df_copy = df.copy()
            df_copy['index'] = df_copy.index
            
            # Sort and set datetime index
            df_sorted = df_copy.sort_values(['hashed_consumer_id', 'adjusted_pmt_created_at'])
            df_sorted.set_index('adjusted_pmt_created_at', inplace=True)
            
            # Calculate rolling counts
            result = df_sorted.groupby('hashed_consumer_id')[column].rolling(
                window=window,
                min_periods=1,
                closed='left'
            ).count()
            
            # Reset index and map back to original
            result = result.reset_index(level=0, drop=True)
            result.index = df_sorted['index']
            result = result.reindex(df_copy['index'])
            
            return result.fillna(0)
            
        except Exception as e:
            logging.error(f"Error in _calculate_rolling_count: {str(e)}")
            logging.error(f"Column: {column}, Window: {window}")
            logging.error(traceback.format_exc())
            # Return zeros on error as fallback
            return pd.Series(0, index=df.index)
    
    def _calculate_rolling_sum(self, df, column, window):
        """Enhanced rolling sum calculator with index preservation"""
        try:
            # Create copy to preserve original data
            df_copy = df.copy()
            df_copy['index'] = df_copy.index
            
            # Sort and set datetime index
            df_sorted = df_copy.sort_values(['hashed_consumer_id', 'adjusted_pmt_created_at'])
            df_sorted.set_index('adjusted_pmt_created_at', inplace=True)
            
            # Calculate rolling sums
            result = df_sorted.groupby('hashed_consumer_id')[column].rolling(
                window=window,
                min_periods=1,
                closed='left'
            ).sum()
            
            # Reset index and map back to original
            result = result.reset_index(level=0, drop=True)
            result.index = df_sorted['index']
            result = result.reindex(df_copy['index'])
            
            return result.fillna(0)
            
        except Exception as e:
            logging.error(f"Error in _calculate_rolling_sum: {str(e)}")
            logging.error(f"Column: {column}, Window: {window}")
            logging.error(traceback.format_exc())
            # Return zeros on error as fallback
            return pd.Series(0, index=df.index)
    
    def _calculate_velocity(self, df, column, window):
        """Enhanced velocity calculator with error handling"""
        try:
            count = self._calculate_rolling_count(df, column, window)
            window_hours = pd.Timedelta(window).total_seconds() / 3600
            velocity = count / window_hours
            return velocity.fillna(0)
        except Exception as e:
            logging.error(f"Error in _calculate_velocity: {str(e)}")
            logging.error(f"Column: {column}, Window: {window}")
            logging.error(traceback.format_exc())
            return pd.Series(0, index=df.index)    

    def create_device_features(self, df):
        """Enhanced device features (25.0% importance)"""
        try:
            # Calculate sophisticated device metrics
            device_stats = df.groupby('device').agg({
                'fraud_flag': ['mean', 'count', 'std'],
                'amount': ['mean', 'std', 'max'],
                'hashed_consumer_id': 'nunique'
            })
            
            # Create amount buckets
            df['amount_bucket'] = pd.qcut(df['amount'], q=5, duplicates='drop')
            
            features = {
                # Time-based patterns
                'device_count_1h': self._calculate_rolling_count(df, 'device', '1H'),
                'device_count_6h': self._calculate_rolling_count(df, 'device', '6H'),
                'device_count_24h': self._calculate_rolling_count(df, 'device', '24H'),
                
                # Risk metrics
                'device_hour_risk': df.groupby(['device', 
                    df['adjusted_pmt_created_at'].dt.hour])['fraud_flag'].transform('mean'),
                'device_amount_risk': df.groupby(['device', 
                    'amount_bucket'])['fraud_flag'].transform('mean'),
                
                # Velocity metrics
                'device_velocity_1h': self._calculate_velocity(df, 'device', '1H'),
                'device_velocity_24h': self._calculate_velocity(df, 'device', '24H'),
                'device_velocity_ratio': (
                    self._calculate_velocity(df, 'device', '1H') /
                    (self._calculate_velocity(df, 'device', '24H') + 1)
                ),
                
                # User patterns
                'device_user_ratio': (
                    df.groupby('device')['hashed_consumer_id'].transform('nunique') /
                    df.groupby('device')['payment_id'].transform('count')
                )
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error in create_device_features: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def create_account_age_features(self, df):
            """Enhanced account age features (49.7% importance)"""
            try:
                # Calculate base age in hours
                df['account_age_hours'] = (
                    pd.to_datetime(df['adjusted_pmt_created_at']) - 
                    pd.to_datetime(df['adjusted_acc_created_at'])
                ).dt.total_seconds() / 3600
                
                # Create sophisticated time-based features
                features = {
                    # Core age features
                    'is_very_new_account': df['account_age_hours'] < 1,
                    'is_new_account': df['account_age_hours'] < 24,
                    'is_week_old': df['account_age_hours'] < 168,
                    
                    # Enhanced risk scoring
                    'account_first_hour_risk': np.exp(-df['account_age_hours']),
                    'account_first_day_risk': np.exp(-df['account_age_hours'] / 24),
                    'account_first_week_risk': np.exp(-df['account_age_hours'] / 168),
                    
                    # Hour-based patterns
                    'creation_hour_sin': np.sin(2 * np.pi * df['adjusted_acc_created_at'].dt.hour / 24),
                    'creation_hour_cos': np.cos(2 * np.pi * df['adjusted_acc_created_at'].dt.hour / 24),
                    
                    # Transaction patterns
                    'first_day_tx_count': self._calculate_rolling_count(df, 'payment_id', '24H'),
                    'first_hour_tx_count': self._calculate_rolling_count(df, 'payment_id', '1H'),
                    
                    # Amount ratios
                    'account_age_amount_ratio': df['amount'] / (df['account_age_hours'] + 1),
                    'hourly_amount_ratio': df['amount'] / (self._calculate_rolling_sum(df, 'amount', '1H') + 1)
                }
                
                # Add age bucket risk scoring
                try:
                    df['age_bucket'] = pd.qcut(df['account_age_hours'], q=10, duplicates='drop')
                    features['age_bucket_risk'] = df.groupby('age_bucket')['fraud_flag'].transform('mean')
                except Exception as e:
                    logging.warning(f"Error in age bucket calculation: {str(e)}")
                    features['age_bucket_risk'] = features['account_first_day_risk']
                
                return features
                
            except Exception as e:
                logging.error(f"Error in create_account_age_features: {str(e)}")
                logging.error(traceback.format_exc())
                raise
    
    def create_velocity_features(self, df):
        """Enhanced velocity features (15.0% importance)"""
        try:
            # Calculate base counts
            tx_1h = self._calculate_rolling_count(df, 'payment_id', '1H')
            tx_6h = self._calculate_rolling_count(df, 'payment_id', '6H')
            tx_24h = self._calculate_rolling_count(df, 'payment_id', '24H')
            
            features = {
                # Transaction counts
                'tx_count_1h': tx_1h,
                'tx_count_6h': tx_6h,
                'tx_count_24h': tx_24h,
                
                # Velocity changes
                'tx_velocity_change': tx_1h / (tx_6h/6 + 1),
                'tx_velocity_change_24h': tx_1h / (tx_24h/24 + 1),
                
                # Amount acceleration
                'amount_acceleration': (
                    self._calculate_rolling_sum(df, 'amount', '1H') /
                    (self._calculate_rolling_sum(df, 'amount', '6H')/6 + 1)
                ),
                
                # Compound metrics
                'tx_amount_ratio': (
                    self._calculate_rolling_sum(df, 'amount', '1H') /
                    (tx_1h + 1)
                )
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error in create_velocity_features: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def create_base_features(self, df):
        try:
            # Store original index
            original_index = df.index
            all_features = {}
            
            # Create each feature set with proper index handling
            for feature_type, create_func in [
                ('account', self.create_account_age_features),
                ('device', self.create_device_features),
                ('velocity', self.create_velocity_features),
                ('identity', self.create_identity_features),
                ('amount', self.create_amount_features)
            ]:
                try:
                    features = create_func(df)
                    # Ensure features have correct index
                    features = pd.DataFrame(features, index=original_index)
                    all_features.update({f"{feature_type}_{k}": v for k, v in features.items()})
                except Exception as e:
                    logging.error(f"Error creating {feature_type} features: {str(e)}")
                    continue
            
            # Create DataFrame and ensure index
            feature_df = pd.DataFrame(all_features, index=original_index)
            
            # Convert numeric columns
            for col in feature_df.columns:
                try:
                    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                except:
                    continue
            
            # Fill NaN values
            feature_df = feature_df.fillna(0)
            
            # Verify index consistency
            if not feature_df.index.equals(original_index):
                raise ValueError("Feature DataFrame index mismatch")
            
            return feature_df
            
        except Exception as e:
            logging.error(f"Error in create_base_features: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def create_interaction_features(self, df, base_features):
        """Create interaction features with proper error handling"""
        try:
            features = {}
            
            # Log available features
            logging.info(f"Available base features: {base_features.columns.tolist()}")
            
            # Account Age x Amount interactions (49.7% x 4.0%)
            if all(col in base_features.columns for col in ['account_first_day_risk', 'amount_amount_zscore']):
                features['interaction_age_amount'] = (
                    base_features['account_first_day_risk'] * 
                    base_features['amount_amount_zscore'].clip(lower=0)
                )
            
            if all(col in base_features.columns for col in ['account_is_very_new_account', 'amount_is_high_amount']):
                features['interaction_new_amount'] = (
                    base_features['account_is_very_new_account'].astype(int) * 
                    base_features['amount_is_high_amount'].astype(int)
                )
            
            # Device x Velocity interactions (25.0% x 15.0%)
            if all(col in base_features.columns for col in ['device_hour_risk', 'velocity_tx_count_1h']):
                features['interaction_device_velocity'] = (
                    base_features['device_hour_risk'] * 
                    base_features['velocity_tx_count_1h']
                )
            
            # Identity x Account Age interactions (5.3% x 49.7%)
            if all(col in base_features.columns for col in ['identity_email_match', 'account_first_day_risk']):
                features['interaction_identity_age'] = (
                    base_features['identity_email_match'] * 
                    base_features['account_first_day_risk']
                )
            
            # Create DataFrame with interaction features
            if features:
                feature_df = pd.DataFrame(features, index=base_features.index)
                logging.info(f"Created {len(features)} interaction features")
                return feature_df
            else:
                logging.warning("No interaction features could be created")
                return pd.DataFrame(index=base_features.index)
            
        except Exception as e:
            logging.error(f"Error in create_interaction_features: {str(e)}")
            logging.error(traceback.format_exc())
            return pd.DataFrame(index=base_features.index)
          
    def create_identity_features(self, df):
        """Create identity verification features"""
        return {
            'email_match': (df['hashed_buyer_email'] == df['hashed_consumer_email']).astype(int),
            'phone_match': (df['hashed_buyer_phone'] == df['hashed_consumer_phone']).astype(int),
            'email_reuse_count': df.groupby('hashed_buyer_email')['payment_id'].transform('count'),
            'phone_reuse_count': df.groupby('hashed_buyer_phone')['payment_id'].transform('count'),
            'identity_risk_score': self._calculate_identity_risk(df)
        }

    def create_amount_features(self, df):
        """Create amount pattern features"""
        amount_mean = df.groupby('hashed_consumer_id')['amount'].transform('mean')
        amount_std = df.groupby('hashed_consumer_id')['amount'].transform('std')
        
        return {
            'amount_zscore': (df['amount'] - amount_mean) / amount_std.fillna(1),
            'is_round_amount': (df['amount'] % 1000 == 0).astype(int),
            'amount_to_mean_ratio': df['amount'] / amount_mean.fillna(df['amount']),
            'amount_percentile': df.groupby('hashed_consumer_id')['amount'].transform(lambda x: x.rank(pct=True)),
            'is_high_amount': (df['amount'] > amount_mean + 2 * amount_std).astype(int)
        }

    def _calculate_identity_risk(self, df):
        email_risk = 1 - df.groupby('hashed_buyer_email')['fraud_flag'].transform('mean')
        phone_risk = 1 - df.groupby('hashed_buyer_phone')['fraud_flag'].transform('mean')
        return (email_risk + phone_risk) / 2

    def _validate_features(self, feature_df):
        """
        Validate feature DataFrame for issues with proper type checking
        """
        try:
            # Check for nulls
            null_cols = feature_df.columns[feature_df.isnull().any()].tolist()
            if null_cols:
                logging.warning(f"Null values found in features: {null_cols}")
                feature_df.fillna(0, inplace=True)
            
            # Check for infinities only in numeric columns
            numeric_cols = feature_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
            if len(numeric_cols) > 0:
                inf_mask = np.isinf(feature_df[numeric_cols])
                inf_cols = numeric_cols[inf_mask.any()].tolist()
                if inf_cols:
                    logging.warning(f"Infinite values found in features: {inf_cols}")
                    feature_df[inf_cols] = feature_df[inf_cols].replace([np.inf, -np.inf], 0)
            
            # Check for categorical columns
            categorical_cols = feature_df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if feature_df[col].nunique() > 1000:
                    logging.warning(f"High cardinality in categorical feature: {col}")
                    
            return feature_df
            
        except Exception as e:
            logging.error(f"Error in _validate_features: {str(e)}")
            logging.error(traceback.format_exc())
            raise

# Part 3: Fraud Detection System Class
class EnhancedFraudDetectionSystem:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.optimal_threshold = None
        self.feature_engineering = FeatureEngineering()
        self.best_params = None
        
    def prepare_features(self, df):
        """Prepare features with index preservation"""
        try:
            # Create base features
            base_features = self.feature_engineering.create_base_features(df)
            if base_features.empty:
                raise ValueError("Failed to create base features")
                
            # Ensure base features have same index as input DataFrame
            base_features.index = df.index
            
            # Create interaction features
            interaction_features = self.feature_engineering.create_interaction_features(df, base_features)
            interaction_features.index = df.index
            
            # Combine features
            if not interaction_features.empty:
                feature_df = pd.concat([base_features, interaction_features], axis=1)
            else:
                feature_df = base_features
                logging.warning("Using only base features due to interaction feature creation failure")
            
            # Apply feature weights
            for category, weight in self.feature_engineering.feature_weights.items():
                cols = [col for col in feature_df.columns if col.startswith(category)]
                if cols:
                    feature_df[cols] = feature_df[cols] * weight
            
            # Verify final feature matrix shape
            if len(feature_df) != len(df):
                raise ValueError(f"Feature matrix shape {len(feature_df)} does not match input data shape {len(df)}")
            
            # Get column types
            numerical_columns = feature_df.select_dtypes(include=['int64', 'float64']).columns
            categorical_columns = feature_df.select_dtypes(include=['object']).columns
            
            logging.info(f"Final feature matrix shape: {feature_df.shape}")
            return feature_df, numerical_columns, categorical_columns
            
        except Exception as e:
            logging.error(f"Error in prepare_features: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def validate_and_clean_data(self, df):
        """
        Validate and clean input data
        """
        try:
            print("\nValidating and cleaning data...")
            
            # Store original shape
            original_shape = df.shape
            
            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.any():
                print(f"\nNull values found in columns:")
                for col in null_counts[null_counts > 0].index:
                    print(f"{col}: {null_counts[col]} nulls")
            
            # Handle fraud_flag specifically
            if 'fraud_flag' in df.columns:
                # Check null values in fraud_flag
                fraud_nulls = df['fraud_flag'].isnull().sum()
                if fraud_nulls > 0:
                    print(f"\nFound {fraud_nulls} null values in fraud_flag")
                    # Fill nulls with 0 (non-fraud) as default
                    df['fraud_flag'] = df['fraud_flag'].fillna(0)
                
                # Ensure binary classification
                df['fraud_flag'] = df['fraud_flag'].astype(int)
                
                # Verify unique values
                unique_vals = df['fraud_flag'].unique()
                print(f"\nUnique values in fraud_flag after cleaning: {unique_vals}")
            
            # Fill missing values for other important columns
            df['device'] = df['device'].fillna('Unknown')
            df['amount'] = df['amount'].fillna(df['amount'].median())
            
            # Convert timestamps
            for col in ['adjusted_pmt_created_at', 'adjusted_acc_created_at']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Print cleaning summary
            print(f"\nData cleaning summary:")
            print(f"Original shape: {original_shape}")
            print(f"Final shape: {df.shape}")
            
            return df
        
        except Exception as e:
            logging.error(f"Error in validate_and_clean_data: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def optimize_and_train(self, X_data, y):
        """Enhanced model optimization and training with validation set"""
        try:
            X, numerical_columns, categorical_columns = X_data
            
            # Create train and validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # Create preprocessor
            transformers = []
            if len(numerical_columns) > 0:
                transformers.append(('num', StandardScaler(), numerical_columns))
            if len(categorical_columns) > 0:
                transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_columns))
            preprocessor = ColumnTransformer(transformers=transformers)

            # Define sampling strategy - 15% of majority class
            sampling_strategy = {1: int(sum(y_train == 0) * 0.15)}

            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('sampling', SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)),
                ('classifier', xgb.XGBClassifier(
                    objective='binary:logistic',
                    tree_method='hist',
                    eval_metric=['auc', 'aucpr'],
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    min_child_weight=3,
                    gamma=0.2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
                    random_state=42
                ))
            ])

            param_dist = {
                'classifier__learning_rate': uniform(0.01, 0.1),
                'classifier__max_depth': [3, 4, 5],
                'classifier__min_child_weight': [1, 3, 5],
                'classifier__gamma': [0.1, 0.2, 0.3],
                'classifier__subsample': [0.6, 0.7, 0.8],
                'classifier__colsample_bytree': [0.6, 0.7, 0.8],
                'classifier__reg_alpha': [0, 0.1, 0.5],
                'classifier__reg_lambda': [0, 0.1, 0.5]
            }

            def custom_fraud_score(y_true, y_pred):
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred)
                f2 = fbeta_score(y_true, y_pred, beta=2)
                return 0.6 * precision + 0.2 * recall + 0.2 * f2

            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=20,
                scoring=make_scorer(custom_fraud_score),
                n_jobs=1,
                cv=3,
                verbose=2,
                error_score='raise'
            )

            # Fit with validation data
            preprocessor.fit(X_train)
            X_val_transformed = preprocessor.transform(X_val)
            
            fit_params = {
                'classifier__eval_set': [(X_val_transformed, y_val)]
            }
            
            random_search.fit(X_train, y_train, **fit_params)

            self.model = random_search.best_estimator_
            self.preprocessor = self.model.named_steps['preprocessor']
            self.best_params = random_search.best_params_
            
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            self.optimal_threshold = self._optimize_threshold(y_val, val_pred_proba)
            
            return self.model

        except Exception as e:
            logging.error(f"Error in optimize_and_train: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    # def optimize_and_train(self, X_data, y):
    #     try:
    #         X, numerical_columns, categorical_columns = X_data
            
    #         # Train-validation split
    #         X_train, X_val, y_train, y_val = train_test_split(
    #             X, y, test_size=0.2, stratify=y, random_state=42
    #         )
            
    #         # Preprocessor
    #         transformers = []
    #         if len(numerical_columns) > 0:
    #             transformers.append(('num', StandardScaler(), numerical_columns))
    #         if len(categorical_columns) > 0:
    #             transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_columns))
    #         preprocessor = ColumnTransformer(transformers=transformers)

    #         # Reduced sampling ratio to 10% 
    #         sampling_strategy = {1: int(sum(y_train == 0) * 0.1)}

    #         # Pipeline with balanced class weights
    #         pipeline = ImbPipeline([
    #             ('preprocessor', preprocessor),
    #             ('sampling', SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)),
    #             ('classifier', xgb.XGBClassifier(
    #                 objective='binary:logistic',
    #                 tree_method='hist',
    #                 eval_metric=['auc', 'aucpr', 'error'],
    #                 scale_pos_weight=3,  # Increased weight for minority class
    #                 random_state=42
    #             ))
    #         ])

    #         # Broader parameter search space
    #         param_dist = {
    #             'classifier__learning_rate': uniform(0.001, 0.1),
    #             'classifier__n_estimators': randint(100, 500),
    #             'classifier__max_depth': randint(3, 8),
    #             'classifier__min_child_weight': uniform(1, 10),
    #             'classifier__subsample': uniform(0.6, 0.4),  # Max 1.0
    #             'classifier__colsample_bytree': uniform(0.6, 0.4),  # Max 1.0
    #             'classifier__gamma': uniform(0, 0.5),
    #             'classifier__reg_alpha': uniform(0, 1),
    #             'classifier__reg_lambda': uniform(0, 1),
    #             'classifier__scale_pos_weight': [2, 3, 4, 5]
    #         }

    #         # Custom scorer with balanced metrics
    #         def custom_fraud_score(y_true, y_pred):
    #             precision = precision_score(y_true, y_pred, zero_division=0)
    #             recall = recall_score(y_true, y_pred)
    #             f2 = fbeta_score(y_true, y_pred, beta=2)
    #             return 0.4 * precision + 0.4 * recall + 0.2 * f2  # Equal weight to precision/recall

    #         random_search = RandomizedSearchCV(
    #             pipeline,
    #             param_distributions=param_dist,
    #             n_iter=50,  # Increased iterations
    #             scoring=make_scorer(custom_fraud_score),
    #             n_jobs=1,
    #             cv=3,
    #             verbose=2,
    #             error_score='raise'
    #         )

    #         # Fit with validation data
    #         preprocessor.fit(X_train)
    #         X_val_transformed = preprocessor.transform(X_val)
            
    #         fit_params = {
    #             'classifier__eval_set': [(X_val_transformed, y_val)]
    #         }
            
    #         random_search.fit(X_train, y_train, **fit_params)

    #         self.model = random_search.best_estimator_
    #         self.preprocessor = self.model.named_steps['preprocessor']
    #         self.best_params = random_search.best_params_
            
    #         # Lower threshold optimization weights
    #         val_pred_proba = self.model.predict_proba(X_val)[:, 1]
    #         precisions, recalls, thresholds = precision_recall_curve(y_val, val_pred_proba)
    #         f2_scores = [fbeta_score(y_val, (val_pred_proba >= t).astype(int), beta=2) 
    #                     for t in thresholds]
    #         # Convert lists to numpy arrays first
    #         precisions = np.array(precisions[:-1])  # Exclude last element to match thresholds
    #         recalls = np.array(recalls[:-1])
    #         f2_scores = np.array(f2_scores)

            
    #         # Find threshold that balances precision/recall
    #         combined_scores = 0.4 * precisions + 0.4 * recalls + 0.2 * f2_scores
    #         optimal_idx = np.argmax(combined_scores)
    #         self.optimal_threshold = thresholds[optimal_idx]
    #         # optimal_idx = np.argmax(0.4 * np.array(precisions[:-1]) + 0.4 * np.array(recalls[:-1]) + 0.2 * np.array(f2_scores))
    #         # self.optimal_threshold = thresholds[optimal_idx]
            
    #         return self.model

    #     except Exception as e:
    #         logging.error(f"Error in optimize_and_train: {str(e)}")
    #         logging.error(traceback.format_exc())
    #         raise

    def _optimize_threshold(self, y_true, y_pred_proba):
        """
        Enhanced threshold optimization with business costs
        """
        # Define business costs
        costs = {
            'false_positive': 100,   # Cost of wrongly blocking transaction
            'false_negative': 1000,  # Cost of missing fraud
            'true_positive': -500    # Benefit of catching fraud
        }
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        total_costs = []
        f2_scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate total cost
            total_cost = (
                fp * costs['false_positive'] +
                fn * costs['false_negative'] +
                tp * costs['true_positive']
            )
            
            total_costs.append(total_cost)
            f2_scores.append(fbeta_score(y_true, y_pred, beta=2))
        
        # Find optimal threshold considering both cost and F2 score
        normalized_costs = np.array(total_costs) / max(abs(min(total_costs)), abs(max(total_costs)))
        normalized_f2 = np.array(f2_scores)
        
        # Combine metrics with weights
        combined_score = -normalized_costs + normalized_f2
        optimal_idx = np.argmax(combined_score)
        
        return thresholds[optimal_idx]

    def evaluate_and_save_results(self, X_data, y, dataset_label='Full'):
        """Evaluate model and save results"""
        X, _, _ = X_data
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'f2': fbeta_score(y, y_pred, beta=2),
            'auc_roc': roc_auc_score(y, y_pred_proba),
            'auc_pr': average_precision_score(y, y_pred_proba),
            'optimal_threshold': self.optimal_threshold
        }

        # Create output directories
        os.makedirs('figures_XGB_1', exist_ok=True)
        os.makedirs('outputs_XGB_1', exist_ok=True)

        # Save metrics
        with open(f'outputs_XGB_1/{dataset_label.lower()}_metrics.txt', 'w') as f:
            f.write(f"Model Performance Metrics on {dataset_label} Data:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")

        # Save plots
        self._save_plots(y, y_pred, y_pred_proba, metrics, dataset_label)

        return metrics

    def _save_plots(self, y_true, y_pred, y_pred_proba, metrics, dataset_label):
        """Save evaluation plots"""
        # ROC Curve
        plt.figure(figsize=(10, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["auc_roc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({dataset_label} Data)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures_XGB_1/{dataset_label.lower()}_roc_curve.png')
        plt.close()

        # Precision-Recall Curve
        plt.figure(figsize=(10, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.plot(recall, precision, label=f'AP = {metrics["auc_pr"]:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({dataset_label} Data)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures_XGB_1/{dataset_label.lower()}_precision_recall_curve.png')
        plt.close()

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix ({dataset_label} Data)')
        plt.tight_layout()
        plt.savefig(f'figures_XGB_1/{dataset_label.lower()}_confusion_matrix.png')
        plt.close()

        # Feature Importance Plot
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            importances = self.model.named_steps['classifier'].feature_importances_

            # Get feature names after preprocessing
            preprocessor = self.model.named_steps['preprocessor']
            feature_names = []

            # Numerical features
            if 'num' in preprocessor.named_transformers_:
                num_transformer = preprocessor.named_transformers_['num']
                num_features = num_transformer.get_feature_names_out()
                feature_names.extend(num_features.tolist())

            # Categorical features
            if 'cat' in preprocessor.named_transformers_:
                cat_transformer = preprocessor.named_transformers_['cat']
                if hasattr(cat_transformer, 'get_feature_names_out'):
                    try:
                        cat_features = cat_transformer.get_feature_names_out()
                        feature_names.extend(cat_features.tolist())
                    except:
                        logging.warning("Categorical transformer is not fitted. Skipping categorical feature names.")
                        # Generate placeholder names if needed
                        num_missing = len(importances) - len(feature_names)
                        cat_features = [f"cat_{i}" for i in range(num_missing)]
                        feature_names.extend(cat_features)
                else:
                    # For older versions of scikit-learn
                    cat_features = cat_transformer.get_feature_names()
                    feature_names.extend(cat_features)
            else:
                logging.info("No categorical features to process.")

            # Ensure feature_names matches the length of importances
            if len(feature_names) != len(importances):
                num_missing = len(importances) - len(feature_names)
                feature_names.extend([f'feature_{i}' for i in range(num_missing)])

            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            # Plot top 20 features
            plt.figure(figsize=(12, 6))
            sns.barplot(data=importance_df.head(20), x='importance', y='feature')
            plt.title(f'Top 20 Feature Importance ({dataset_label} Data)')
            plt.tight_layout()
            plt.savefig(f'figures_XGB_1/{dataset_label.lower()}_feature_importance.png')
            plt.close()

    def save_model(self, filepath):
        """Save the trained model and its components"""
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'optimal_threshold': self.optimal_threshold,
            'best_params': self.best_params
        }
        joblib.dump(model_data, filepath)
        logging.info(f"Model saved to {filepath}")

    def validate_model(self, X_data, y):
        """Validate model performance on different segments"""
        try:
            X, _, _ = X_data
            predictions = self.model.predict_proba(X)[:, 1]
            
            # Validate different amount ranges
            amount_ranges = [(0, 1000), (1000, 5000), (5000, float('inf'))]
            for min_amt, max_amt in amount_ranges:
                mask = (X['amount'] >= min_amt) & (X['amount'] < max_amt)
                if mask.any():
                    score = roc_auc_score(y[mask], predictions[mask])
                    logging.info(f"AUC for amount range {min_amt}-{max_amt}: {score:.4f}")
                    
        except Exception as e:
            logging.error(f"Error in validate_model: {str(e)}")


def main():
   """Main execution function"""
   try:
       # Initialize detector
       detector = EnhancedFraudDetectionSystem()

       # Load and preprocess data
       print("Loading and preprocessing data...")
       df = pd.read_csv('data_scientist_fraud_20241009.csv')

       # Validate input data
       if df.empty:
           raise ValueError("Empty input DataFrame")

       # Clean and validate data
       df = detector.validate_and_clean_data(df)

       # Convert timestamp columns to datetime
       timestamp_columns = ['adjusted_pmt_created_at', 'adjusted_acc_created_at']
       for col in timestamp_columns:
           if col not in df.columns:
               raise ValueError(f"Missing required column: {col}")
           df[col] = pd.to_datetime(df[col])
           
       print(f"Timestamps converted: {df[timestamp_columns[0]].dtype}")

       # Ensure required columns exist
       required_columns = [
           'payment_id', 'device', 'amount', 'fraud_flag',
           'hashed_consumer_id', 'hashed_buyer_email', 'hashed_consumer_email',
           'hashed_buyer_phone', 'hashed_consumer_phone'
       ]
       missing_columns = [col for col in required_columns if col not in df.columns]
       if missing_columns:
           raise ValueError(f"Missing required columns: {missing_columns}")
           
       print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

       # Prepare features
       print("\nPreparing features...")
       X_data = detector.prepare_features(df)
       y = df['fraud_flag']

       # Split data
       X, numerical_columns, categorical_columns = X_data
       X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
       X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

       # Train model
       print("\nTraining and optimizing model...")
       detector.optimize_and_train(
           (X_train, numerical_columns, categorical_columns), y_train)

       # Evaluate on test set
       print("\nEvaluating model...")
       metrics = detector.evaluate_and_save_results(
           (X_test, numerical_columns, categorical_columns), 
           y_test,
           dataset_label='Test'
       )

       # Print final metrics
       print("\nFinal Model Performance:")
       for metric, value in metrics.items():
           print(f"{metric}: {value:.4f}")

       # Save high-risk predictions
       predictions = detector.model.predict_proba(X_test)[:, 1]
       high_risk_df = pd.DataFrame({
           'payment_id': df['payment_id'].iloc[X_test.index],
           'fraud_probability': predictions,
           'amount': df['amount'].iloc[X_test.index],
           'hashed_consumer_id': df['hashed_consumer_id'].iloc[X_test.index]
       })
       
       # Filter high-risk transactions
       high_risk_df = high_risk_df[
           high_risk_df['fraud_probability'] >= detector.optimal_threshold
       ]
       
       # Save high-risk transactions
       high_risk_df.to_csv('outputs_XGB_1/high_risk_transactions.csv', index=False)
       print(f"\nIdentified {len(high_risk_df)} high-risk transactions")

       # Save model
       print("\nSaving model...")
       detector.save_model('outputs_XGB_1/fraud_detection_model.pkl')

       print("\nAnalysis completed successfully!")

   except Exception as e:
       print(f"Error during execution: {str(e)}")
       logging.error(f"Error during execution: {str(e)}")
       raise

if __name__ == "__main__":
   main()
