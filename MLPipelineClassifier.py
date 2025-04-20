# optuna paramaterized search tuning/cv
# Scikit pipeline, XGBoost, LightGBM, (catboost, gpu accelerated mlpr through torch two layer nn, and futures later)
# Additional Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import (PredefinedSplit, GridSearchCV, train_test_split, cross_val_score,
                                     KFold, StratifiedKFold)
import lightgbm as lgb
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
# import xgboost as xgb
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, log_loss)
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
import optuna
from optuna.samplers import TPESampler
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# for logistic regression statistics, if you want it
import statsmodels.api as sm
from Numeric_Check import datetime_normalization
# Working on implementation of catboost
# from catboost import CatBoostClassifier, Pool
import gc


"""
This is an implementation of an end-to-end regression machine learning pipeline (No classification)
This is an implementation of an end-to-end regression machine learning pipeline (No classification)

To use, look at the inputs to the example MLP implementation here

def shallowMachineLearningPipeline(df, test_df, ordinal_columns, categorical_columns, 
                                   numerical_columns, target_col, id_columns=None, 
                                   datetime_columns=None, output_dir=None):

Specify the column types as listed above, with ordinal/categoricals columns in a categorical/object dtype
and numericals as a numerical dtype (like int, float). Target_col must be 
ID_Columns are not used for training, but reattached to predictions in the test set (not used at all for training)

Missings in numerical, dataset differences, and column creation from OHE/Missing indication all handled internally

Only takes in Pandas Dataframes, for best practices, presplit a dataframe
a test_df and a training df (list as just df). Without a test_df, predictions and evaluation metrics 
will not be produced. The training dataframe will be split into training/validation sets for CV, if
this is not wanted behavior, PredefinedSplits are introduced in a later build.
PredefinedSplits cannot be used for Optuna (from what I recall)

A Numeric_Check file is also referenced (In this case, just forces for datetime normalization on datetime variables)

No proper inbuilt logging or error checking currently built in.

For loading test models, please use the proper FittedEncoders, outputted in a joblib file
    Stores the training numerical missing input statistic/strategy (like median imputation, etc.)
    Training Scalers, known/unknown columns, column names post numerical filling and OHE column creation
    and more...
"""


# Thread Limit Control
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


# Extreme thread limit control (to test)
# The packages are likely engaging these packages at the end of studies.
# os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
# os.environ["NUMEXPR_NUM_THREADS"] = "2"


# Warning Suppression
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='No further splits with positive gain.*')
warnings.filterwarnings('ignore', category=UserWarning, message='Found `n_estimators` in params.*')


def scale_with_na(df, numerical_columns):
    """
    Scales with sklearn's StandardScaler and returns the Scaler
    Use on a dataframe with specified numerical_columns in a list
        I originally wrote this to preserve general columns (not just numerical) for some specific models
        But then I wrote some other stuff and categorical/ordinal columns have their own scaling methods with masks
        and I added imputation + missing-indicator columns for numerical columns, which I believe is the best general
        way to handle missings in a black box
    """


    df = df.copy()
    scalers = {}

    for col in numerical_columns:
        scaler = StandardScaler()
        # Reshape to 2D array as required by sklearn
        values = df[col].values.reshape(-1, 1)
        df[col] = scaler.fit_transform(values).ravel()
        scalers[col] = scaler

    return df, scalers


# Predefined split function creation, reproducible splits without k-fold CV using sklearn's model_selection
# Faster/More reproducible than CV, but it'll overfit on validation (especially with smaller datasets) with tuner
# Move to Utilities file, courtesy of Professor Yuxiao Huang, GWU
def get_train_val_ps(X_train, y_train, X_val, y_val):
    # Combine the feature matrix in the training and validation data
    X_train_val = np.vstack((X_train, X_val))

    # Combine the target vector in the training and validation data
    y_train_val = np.vstack((y_train.reshape(-1, 1), y_val.reshape(-1, 1))).reshape(-1)

    # Get the indices of training and validation data
    train_val_idxs = np.append(np.full(X_train.shape[0], -1), np.full(X_val.shape[0], 0))

    # The PredefinedSplit
    ps = PredefinedSplit(train_val_idxs)

    return X_train_val, y_train_val, ps


class FeatureNameTracker:
    """
        Tracks and manages feature names across different model types (sklearn, lightgbm, catboost)

        This class maintains mappings between original feature names and their transformed versions
        for different model types, handling ordinal, categorical, and numerical features differently

        e.g. ohe for categorical columns in sklearn, le for categoricals in lightgbm, no encoding for catboost
        Really, this is only for the ohe columns in sklearn
        But later, also for missing column indicator for numerical columns

        Attributes:
            ordinal_columns (list): Names of ordinal features
            categorical_columns (list): Names of categorical features
            numerical_columns (list): Names of numerical features
            original_features (dict): Original feature names grouped by type
            transformed_features (dict): Transformed feature names for each model type
            feature_mappings (dict): Mappings between original and transformed feature names

        Example:
            tracker = FeatureNameTracker(
                ordinal_columns=['size', 'rank'],
                categorical_columns=['color', 'gender'],
                numerical_columns=['price', 'num_items']
            )
            tracker.update_sklearn_features(['color_red', 'color_blue'])
    """


    def __init__(self, ordinal_columns, categorical_columns, numerical_columns):
        self.ordinal_columns = ordinal_columns
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        # Store original column names by type
        self.original_features = {
            'ordinal': ordinal_columns,
            'categorical': categorical_columns,
            'numerical': numerical_columns
        }

        # Will store transformed feature names
        self.transformed_features = {
            'sklearn': {
                'ordinal': [],
                'categorical': [],  # Will store ohe column names
                'numerical': []
            },
            'lightgbm': {
                'ordinal': [],
                'categorical': [],  # Will store label encoded column names
                'numerical': []
            },
            'catboost': {
                'ordinal': [],
                'categorical': [],  # Will store base string/categorical dtype column names
                'numerical': []
            }
        }

        # Will store feature mappings
        self.feature_mappings = {
            'sklearn': {},  # e.g., {'color_red': 'color', 'color_blue': 'color'}  due to ohe encoding
            'lightgbm': {},  # e.g., {'color': 'color'} needs separate label encoding from sklearn
            'catboost': {}
        }


    def update_sklearn_features(self, ohe_feature_names):
        # Update sklearn feature names after one-hot encoding
        self.transformed_features['sklearn']['ordinal'] = self.ordinal_columns
        self.transformed_features['sklearn']['numerical'] = self.numerical_columns
        self.transformed_features['sklearn']['categorical'] = ohe_feature_names

        # Update mappings
        for feature in self.ordinal_columns + self.numerical_columns:
            self.feature_mappings['sklearn'][feature] = feature

        # Map OHE features back to original categorical columns
        for ohe_feature in ohe_feature_names:
            # Extract original column name from OHE feature name
            # Assuming format like "color_red", "color_blue" etc.
            original_col = ohe_feature.split('_')[0]
            self.feature_mappings['sklearn'][ohe_feature] = original_col

    def update_lightgbm_features(self):
        # Update LightGBM (1 to 1)
        self.transformed_features['lightgbm'] = {
            'ordinal': self.ordinal_columns,
            'categorical': self.categorical_columns,
            'numerical': self.numerical_columns
        }

        # Direct 1:1 mapping for LightGBM
        for feature in (self.ordinal_columns +
                        self.categorical_columns +
                        self.numerical_columns):
            self.feature_mappings['lightgbm'][feature] = feature

    def update_catboost_features(self):
        # Update CatBoost (1 to 1)
        self.transformed_features['catboost'] = {
            'ordinal': self.ordinal_columns,
            'categorical': self.categorical_columns,
            'numerical': self.numerical_columns
        }

        # Direct 1:1 mapping for CatBoost
        for feature in (self.ordinal_columns +
                        self.categorical_columns +
                        self.numerical_columns):
            self.feature_mappings['catboost'][feature] = feature

    def get_feature_names(self, model_type='sklearn'):
        # Get all feature names for a specific model type
        features = self.transformed_features[model_type]
        return (features['ordinal'] +
                features['categorical'] +
                features['numerical'])

    def get_original_feature(self, transformed_feature, model_type='sklearn'):
        # Get original feature name from transformed feature name
        return self.feature_mappings[model_type].get(transformed_feature)

    def get_feature_type(self, feature):
        # Get type of original feature (ordinal/categorical/numerical)
        if feature in self.ordinal_columns:
            return 'ordinal'
        elif feature in self.categorical_columns:
            return 'categorical'
        elif feature in self.numerical_columns:
            return 'numerical'
        return None


# DataPreprocessor Function
def dataPreprocessor(train_df, ordinal_columns, categorical_columns, numerical_columns, target_col,
                     id_columns=None, datetime_columns=None, test_df=None, drop_first=True, random_state=42):
    """
    Creates three datasets optimized for different model types, with optional test set transformation.

    Parameters:
    -----------
    train_df : pandas DataFrame
        Training data to fit preprocessors on
    test_df : pandas DataFrame, optional
        Test data to transform using fitted preprocessors

    Returns:
    --------
    dict containing:
        - sklearn_ready: DataFrame ready for sklearn models (for pipeline ingestion
        - lightgbm_ready: DataFrame ready for LightGBM (Still needs to be converted from df to lightgbm.dataset in
        - catboost_ready: DataFrame ready for CatBoost (No model function call implemented yte
        - fitted_encoders: Dictionary of fitted encoders
    If test_df is provided, each ready dataset will be a dict with 'train' and 'test' keys
    """

    # Track feature names, init
    feature_tracker = FeatureNameTracker(
        ordinal_columns, categorical_columns, numerical_columns
    )

    # Store original IDs if present
    train_ids = train_df[id_columns].copy() if id_columns else None
    test_ids = test_df[id_columns].copy() if test_df is not None and id_columns else None

    # Remove ID columns (for further preprocessing, don't leave it in the ml pipeline)
    train_df = train_df.drop(columns=id_columns) if id_columns else train_df
    test_df = test_df.drop(columns=id_columns) if test_df is not None and id_columns else test_df

    # Store original indexes (for matching back later
    train_index = train_df.index
    test_index = test_df.index if test_df is not None else None

    # Store statistics and encoders
    fitted_encoders = {
        'label_encoders': {},
        'onehot_encoder': None,
        'scalers': {},
        'numerical_stats': {},
        'datetime_features': {},
        'imputers': {},
        'feature_tracker': feature_tracker,
        'OHE_column_name_map' : {}
    }

    # Convert datetime to numerical columns, create day, month, and year columns
    # Add to numerical before generating NA stats
    if datetime_columns:
        needs_month = {}  # Checks on training to see if you need a month column created.
        needs_year = {}

        for datetime_col in datetime_columns:
            # Create mask for NaT values
            nat_mask = train_df[datetime_col].isna()

            # Process non-NaT values
            if not all(nat_mask):  # If there are any non-NaT values
                train_df.loc[~nat_mask, datetime_col] = datetime_normalization(train_df.loc[~nat_mask, datetime_col])
                start_date = min(train_df.loc[~nat_mask, datetime_col])

                day_count_name = f"{datetime_col}_day_count"

                # Initialize as Int64 instead of float
                train_df[day_count_name] = pd.NA

                # Calculate for non-NaT values and convert to Int64
                train_df.loc[~nat_mask, day_count_name] = [(date - start_date).days for date in
                                                           train_df.loc[~nat_mask, datetime_col]]
                train_df[day_count_name] = train_df[day_count_name].astype('Int64')

                if day_count_name not in fitted_encoders['feature_tracker'].numerical_columns:
                    fitted_encoders['feature_tracker'].numerical_columns.append(day_count_name)

                # Track month and year uniqueness only for non-NaT values
                needs_month[datetime_col] = train_df.loc[~nat_mask, datetime_col].dt.month.nunique() > 1
                needs_year[datetime_col] = train_df.loc[~nat_mask, datetime_col].dt.year.nunique() > 1

                fitted_encoders['datetime_features'][datetime_col] = {
                    'start_date': start_date,
                    'needs_month': needs_month,
                    'needs_year': needs_year
                }

                if needs_month[datetime_col]:
                    month_num_name = f"{datetime_col}_month_num"

                    # Initialize with pd.NA for Int64 type
                    train_df[month_num_name] = pd.NA
                    # Calculate for non-NaT values
                    train_df.loc[~nat_mask, month_num_name] = [date.month for date in
                                                               train_df.loc[~nat_mask, datetime_col]]
                    # Convert to Int64
                    train_df[month_num_name] = train_df[month_num_name].astype('Int64')

                    if month_num_name not in fitted_encoders['feature_tracker'].numerical_columns:
                        fitted_encoders['feature_tracker'].numerical_columns.append(month_num_name)

                if needs_year[datetime_col]:
                    year_num_name = f"{datetime_col}_year_num"

                    # Initialize with pd.NA for Int64 type
                    train_df[year_num_name] = pd.NA
                    # Calculate for non-NaT values
                    train_df.loc[~nat_mask, year_num_name] = [date.year for date in
                                                              train_df.loc[~nat_mask, datetime_col]]
                    # Convert to Int64
                    train_df[year_num_name] = train_df[year_num_name].astype('Int64')

                    if year_num_name not in fitted_encoders['feature_tracker'].numerical_columns:
                        fitted_encoders['feature_tracker'].numerical_columns.append(year_num_name)

                # Process test_df similarly
                if test_df is not None:
                    test_nat_mask = test_df[datetime_col].isna()

                    if not all(test_nat_mask):
                        test_df.loc[~test_nat_mask, datetime_col] = datetime_normalization(
                            test_df.loc[~test_nat_mask, datetime_col])

                        # Initialize as Int64
                        test_df[day_count_name] = pd.NA
                        test_df.loc[~test_nat_mask, day_count_name] = [(date - start_date).days for date in
                                                                       test_df.loc[~test_nat_mask, datetime_col]]
                        # Convert to Int64
                        test_df[day_count_name] = test_df[day_count_name].astype('Int64')

                        if needs_month[datetime_col]:
                            test_df[month_num_name] = pd.NA
                            test_df.loc[~test_nat_mask, month_num_name] = [date.month for date in
                                                                           test_df.loc[~test_nat_mask, datetime_col]]
                            test_df[month_num_name] = test_df[month_num_name].astype('Int64')

                        if needs_year[datetime_col]:
                            test_df[year_num_name] = pd.NA
                            test_df.loc[~test_nat_mask, year_num_name] = [date.year for date in
                                                                          test_df.loc[~test_nat_mask, datetime_col]]
                            test_df[year_num_name] = test_df[year_num_name].astype('Int64')

            else:
                # All values are NaT in this column - decide how to handle
                print(
                    f"Warning: All values in {datetime_col} are NaT. Skipping datetime processing for this column.")
                print(f"You should really drop these columns in your dataframes...")


    # Drop the columns now, they can't be used.
    train_df = train_df.drop(columns=datetime_columns) if datetime_columns else train_df
    test_df = test_df.drop(columns=datetime_columns) if test_df is not None and datetime_columns else test_df

    # Store numerical column statistics from training data
    for col in numerical_columns:
        fitted_encoders['numerical_stats'][col] = {
            'mean': train_df[col].mean(),
            'median': train_df[col].median()
        }


    def impute_values(df, col, strategy='median'):
        if strategy == 'mean':
            value = fitted_encoders['numerical_stats'][col]['mean']
        elif strategy == 'median':
            value = fitted_encoders['numerical_stats'][col]['median']
        elif strategy == 'constant':
            value = -999  # or another sentinel value
                          # this will displace the value, but I'm adding a numerical missing indicator col
                          # Biggest issue with using this is it distorts the range post scaling
                          # So basically, don't use this unless you change the value and know your dataset well
        return df[col].fillna(value)


    # Function to process a dataset using fitted transformers
    def process_dataset(df, is_train=True):
        df_copy = df.copy()

        original_index = df_copy.index
        target = df_copy[target_col].copy() if target_col in df_copy.columns else None
        df_copy = df_copy.drop(columns=[target_col]) if target_col in df_copy.columns else df_copy

        # 1. Create sklearn-ready dataset
        sklearn_df = df_copy.copy()

        # Handle numerical columns
        # Have to handle these first, so that when categorical rolls around for the test set they'll be accounted for
        for col in numerical_columns:
            if is_train:
                has_missing = sklearn_df[col].isna().any()
                fitted_encoders['numerical_stats'][col]['has_missing'] = has_missing

            # Create missing indicator column
            if fitted_encoders['numerical_stats'][col]['has_missing']:
                # This will do a simple input if there's no missing indicator in the training set
                # even if there is for the test set, but it'll still run, so be careful
                missing_indicator_name = f"{col}_missing"
                sklearn_df[missing_indicator_name] = sklearn_df[col].isna().astype('int32')

                # Add this to the categorical_columns in our feature_tracker, if it's not already in there.
                if missing_indicator_name not in fitted_encoders['feature_tracker'].categorical_columns:
                    fitted_encoders['feature_tracker'].categorical_columns.append(missing_indicator_name)

            # Missing imputation
            sklearn_df[col] = impute_values(sklearn_df, col)

        # Scale numerical features, after imputing
        if is_train:
            fitted_encoders['scalers']['sklearn'] = {}
            for col in numerical_columns:
                scaler = StandardScaler()
                values = sklearn_df[col].values.reshape(-1, 1)
                sklearn_df[col] = scaler.fit_transform(values).ravel()
                sklearn_df[col] = sklearn_df[col].astype(np.float64)
                fitted_encoders['scalers']['sklearn'][col] = scaler
        else:
            sklearn_scalers = fitted_encoders['scalers']['sklearn']
            for col in numerical_columns:
                values = sklearn_df[col].values.reshape(-1, 1)
                sklearn_df[col] = sklearn_scalers[col].transform(values).ravel()
                sklearn_df[col] = sklearn_df[col].astype(np.float64)

        # Handle ordinals - label encode
        for col in ordinal_columns:
            if is_train:
                # During training, add both MISSING and UNKNOWN
                sklearn_df[col] = sklearn_df[col].fillna('MISSING')

                sklearn_df[col] = sklearn_df[col].map(str)

                # Add UNKNOWN category for future unseen values
                sklearn_df = pd.concat([
                    sklearn_df,
                    pd.DataFrame({col: ['UNKNOWN']}, index=['tmp'])
                ])

                le = LabelEncoder()
                sklearn_df[col] = le.fit_transform(sklearn_df[col])
                fitted_encoders['label_encoders'][col] = le

                # Remove the temporary row
                sklearn_df = sklearn_df.drop(index='tmp')
            else:
                le = fitted_encoders['label_encoders'][col]
                # First handle actual missing values
                sklearn_df[col] = sklearn_df[col].fillna('MISSING')

                # Convert to string (same as in training)
                # Missing this code was causing issues due to mapping everything to 'UNKNOWN'
                sklearn_df[col] = sklearn_df[col].map(str)

                # Then map unseen categories to UNKNOWN
                sklearn_df[col] = sklearn_df[col].map(
                    lambda x: 'UNKNOWN' if x not in le.classes_ else x
                )
                sklearn_df[col] = le.transform(sklearn_df[col])


        # Handle categoricals - one hot encode
        if categorical_columns:
            filtered_categorical_columns = []
            for col in categorical_columns:
                # Skip categorical columns that are missing indicators for numerical columns
                if any(col == f"{num_col}_missing" and fitted_encoders['numerical_stats'][num_col]['has_missing']
                       for num_col in numerical_columns):
                    continue
                filtered_categorical_columns.append(col)

            if filtered_categorical_columns:
                for col in filtered_categorical_columns:
                    sklearn_df[col] = sklearn_df[col].astype(str)
                    sklearn_df[col] = sklearn_df[col].fillna('MISSING')
                    sklearn_df[col] = sklearn_df[col].astype(str).replace('nan', 'MISSING')

                if is_train:
                    # Add UNKNOWN category during training
                    sklearn_df = pd.concat([
                        sklearn_df,
                        pd.DataFrame({col: ['UNKNOWN'] for col in filtered_categorical_columns}, index=['tmp'])
                    ])

                    ohe = OneHotEncoder(sparse_output=False, drop='first' if drop_first else None)
                    cat_encoded = ohe.fit_transform(sklearn_df[filtered_categorical_columns])
                    fitted_encoders['onehot_encoder'] = ohe

                    sklearn_df = sklearn_df.drop(index='tmp')
                    cat_encoded = cat_encoded[:-1]

                    feature_names = ohe.get_feature_names_out(filtered_categorical_columns)

                    # Create a mapping dictionary between original and sanitized names
                    name_mapping = {}
                    sanitized_names = []

                    for name in feature_names:
                        sanitized_name = name.replace('<', '').replace('>', '') \
                            .replace('(', '').replace(')', '').replace(' ', '_') \
                            .replace('=', 'EQUALS').replace('[', '').replace(']', '') \
                            .replace(':', '_').replace('\\', '').replace('/', '')

                        name_mapping[name] = sanitized_name
                        sanitized_names.append(sanitized_name)

                    # Store this mapping in fitted_encoders
                    fitted_encoders['OHE_column_name_map'] = name_mapping

                    cat_encoded_df = pd.DataFrame(
                        cat_encoded, columns=sanitized_names, index=sklearn_df.index
                    )
                else:
                    ohe = fitted_encoders['onehot_encoder']
                    # Map unseen categories to UNKNOWN
                    for col in filtered_categorical_columns:
                        sklearn_df[col] = sklearn_df[col].map(
                            lambda x: x if x in ohe.categories_[filtered_categorical_columns.index(col)] else 'UNKNOWN'
                        )
                    cat_encoded = ohe.transform(sklearn_df[filtered_categorical_columns])

                    feature_names = ohe.get_feature_names_out(filtered_categorical_columns)

                    name_mapping = fitted_encoders['OHE_column_name_map']

                    # Handle any new feature names not in the mapping
                    sanitized_names = []
                    for name in feature_names:
                        if name in name_mapping:
                            sanitized_names.append(name_mapping[name])
                        else:
                            # Apply the same sanitization logic
                            sanitized_name = name.replace('<', '').replace('>', '') \
                                .replace('(', '').replace(')', '').replace(' ', '_') \
                                .replace('=', 'EQUALS').replace('[', '').replace(']', '') \
                                .replace(':', '_').replace('\\', '').replace('/', '')
                            sanitized_names.append(sanitized_name)

                    cat_encoded_df = pd.DataFrame(
                        cat_encoded, columns=sanitized_names, index=sklearn_df.index
                    )

                sklearn_df = sklearn_df.drop(columns=filtered_categorical_columns)
                sklearn_df = pd.concat([sklearn_df, cat_encoded_df], axis=1)

        # Update the sklearn_features with the new columns
        if is_train and categorical_columns:
            feature_names = ohe.get_feature_names_out(filtered_categorical_columns)
            fitted_encoders['feature_tracker'].update_sklearn_features(feature_names)


        # Final conversion for full standardization
        for col in sklearn_df.columns:
            if sklearn_df[col].dtype == 'int32':
                sklearn_df[col] = sklearn_df[col].astype('float64')


        # 2. Create LightGBM-ready dataset
        lgb_df = df_copy.copy()

        # Add missing indicators for LightGBM
        # Don't need to create fitted_encoders again because we did it for sklearn
        for col in numerical_columns:
            if fitted_encoders['numerical_stats'][col]['has_missing']:
                missing_indicator_name = f"{col}_missing"
                lgb_df[missing_indicator_name] = lgb_df[col].isna().astype(int)

            # Handle numerical columns, same imputation approach as sklearn
            lgb_df[col] = impute_values(lgb_df, col)


        # Scale numerical features while preserving NaN values, after imputing
        if is_train:
            lgb_df, lgb_scalers = scale_with_na(lgb_df, numerical_columns)
            fitted_encoders['scalers']['lightgbm'] = lgb_scalers
        else:
            lgb_scalers = fitted_encoders['scalers']['lightgbm']
            for col in numerical_columns:
                values = lgb_df[col].values.reshape(-1, 1)
                lgb_df[col] = lgb_scalers[col].transform(values).ravel()


        # Creating separate label encoders, sklearn's le does not handle categoricals.
        for col in categorical_columns + ordinal_columns:
            if isinstance(lgb_df[col].dtype, pd.CategoricalDtype):
                lgb_df[col] = lgb_df[col].astype(str)

            non_null_mask = lgb_df[col].notna()

            if is_train:
                # Create new LightGBM-specific encoder
                le = LabelEncoder()
                if non_null_mask.any():
                    encoded_values = le.fit_transform(lgb_df.loc[non_null_mask, col])

                    lgb_df.loc[non_null_mask, col] = pd.Series(encoded_values, index=lgb_df[non_null_mask].index,
                                                               dtype="Int64")

                fitted_encoders['label_encoders'][f'lgb_{col}'] = le
            else:
                # Use LightGBM-specific encoder
                le = fitted_encoders['label_encoders'][f'lgb_{col}']
                if non_null_mask.any():
                    non_null_values = lgb_df.loc[non_null_mask, col]
                    is_unseen = ~non_null_values.isin(le.classes_)

                    unseen_indices = non_null_values[is_unseen].index
                    known_indices = non_null_values[~is_unseen].index

                    if len(unseen_indices) > 0:
                        lgb_df.loc[unseen_indices, col] = np.nan

                    if len(known_indices) > 0:
                        encoded_values = le.transform(lgb_df.loc[known_indices, col])

                        lgb_df.loc[known_indices, col] = pd.Series(encoded_values, index=known_indices,
                                                                   dtype="Int64")

            # Ensure the entire column is Int64
            lgb_df[col] = lgb_df[col].astype("Int64")

        if is_train:
            fitted_encoders['feature_tracker'].update_lightgbm_features()

        # 3. Create CatBoost-ready dataset
        catboost_df = df_copy.copy()

        # Add missing indicators for CatBoost
        for col in numerical_columns:
            if fitted_encoders['numerical_stats'][col]['has_missing']:
                missing_indicator_name = f"{col}_missing"
                catboost_df[missing_indicator_name] = catboost_df[col].isna().astype(int)

            catboost_df[col] = impute_values(catboost_df, col)


        # Scale numerical columns
        if is_train:
            catboost_df, catboost_scalers = scale_with_na(catboost_df, numerical_columns)
            fitted_encoders['scalers']['catboost'] = catboost_scalers
        else:
            catboost_scalers = fitted_encoders['scalers']['catboost']
            for col in numerical_columns:
                values = catboost_df[col].values.reshape(-1, 1)
                catboost_df[col] = catboost_scalers[col].transform(values).ravel()

        # Convert ordinals and categoricals to strings
        # Actually, don't do this. It'll fill na types as 'None', the string,
        # and catboost will separate it into a class instead of use it with its NoneType processing
        # for col in ordinal_columns + categorical_columns:
        #     catboost_df[col] = catboost_df[col].astype(str)

        # Convert nulls to None
        catboost_df = catboost_df.where(pd.notna(catboost_df), None)

        if is_train:
            fitted_encoders['feature_tracker'].update_catboost_features()

        if not sklearn_df.index.equals(original_index):
            raise ValueError("sklearn_df index was modified during processing")
        if not lgb_df.index.equals(original_index):
            raise ValueError("lgb_df index was modified during processing")
        if not catboost_df.index.equals(original_index):
            raise ValueError("catboost_df index was modified during processing")

        # Add target back if it exists
        if target is not None:
            sklearn_df[target_col] = target
            lgb_df[target_col] = target
            catboost_df[target_col] = target

        # Add IDs back if they exist, w/ index
        if is_train and train_ids is not None:
            sklearn_df = pd.concat([train_ids, sklearn_df], axis=1)
            lgb_df = pd.concat([train_ids, lgb_df], axis=1)
            catboost_df = pd.concat([train_ids, catboost_df], axis=1)
        elif not is_train and test_ids is not None:
            sklearn_df = pd.concat([test_ids, sklearn_df], axis=1)
            lgb_df = pd.concat([test_ids, lgb_df], axis=1)
            catboost_df = pd.concat([test_ids, catboost_df], axis=1)

        return sklearn_df, lgb_df, catboost_df

    # Call above function produce these datasets, on train
    sklearn_train, lgb_train, catboost_train = process_dataset(train_df, is_train=True)

    # Assert check
    assert sklearn_train.index.equals(train_index)
    assert lgb_train.index.equals(train_index)
    assert catboost_train.index.equals(train_index)

    # Initialize results dictionary
    results = {
        'sklearn_ready': {'train': sklearn_train},
        'lightgbm_ready': {'train': lgb_train},
        'catboost_ready': {'train': catboost_train},
        'fitted_encoders': fitted_encoders
    }

    # Process test data if provided
    if test_df is not None:
        sklearn_test, lgb_test, catboost_test = process_dataset(test_df, is_train=False)

        # Assert check for test dataframes
        assert sklearn_test.index.equals(test_index)
        assert lgb_test.index.equals(test_index)
        assert catboost_test.index.equals(test_index)

        results['sklearn_ready']['test'] = sklearn_test
        results['lightgbm_ready']['test'] = lgb_test
        results['catboost_ready']['test'] = catboost_test

    return results, fitted_encoders['feature_tracker'].categorical_columns


# predone tuner optuna, if gridsearchcv is not supplied
def skLearnTuner(df, target_col, output_dir=None, random_seed=42, n_jobs=20, n_trials=1, tuner_param_ranges=None):
    """
    Runs an Optuna optimization pipeline for sklearn-compatible classification models.

    Parameters:
    -----------
    df : pandas DataFrame
        Preprocessed DataFrame ready for sklearn models
    target_col : str
        Name of the target column
    output_dir : str, optional
        Directory to save optimization results. If None, results aren't saved to disk
    random_seed : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of jobs for parallel processing
    n_trials : int, default=100
        Number of optimization trials for Optuna

    Returns:
    --------
    dict containing:
        - 'best_models': DataFrame with best models and their scores
        - 'study_results': Dict of DataFrames with full optimization results
        - 'best_estimators': Dict of best fitted estimators for each model
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Define model creation functions
    def create_rf():
        return RandomForestClassifier(random_state=random_seed)

    def create_xgb():
        return XGBClassifier(random_state=random_seed)

    models = {
        'rfc': create_rf,
        'xgbc': create_xgb
    }

    if not tuner_param_ranges:
        tuner_param_ranges= {
            'rfc': {
                'n_estimators': (50, 1000, 'int'),
                'max_depth': (3, 25, 'int'),
                'min_samples_split': (2, 10, 'int'),
                'min_samples_leaf': (1, 10, 'int'),
                'max_features': (0.1, 1.0, 'float')
            },
            'xgbc': {
                'max_depth': (3, 20, 'int'),
                'learning_rate': (1e-5, 0.2, 'float_log'),
                'n_estimators': (50, 2000, 'int'),
                'min_child_weight': (1, 20, 'int'),
                'subsample': (0.5, 1.0, 'float'),
                'colsample_bytree': (0.5, 1.0, 'float'),
                'gamma': (1e-4, 3, 'float_log'),
                'reg_alpha': (1e-3, 1.0, 'float_log'),
                'reg_lambda': (1e-3, 1.0, 'float_log'),
                'scale_pos_weight': (1, 10, 'float')  # For class imbalance
            }
        }

    # Store results
    best_models_results = []
    cv_results_dict = {}
    best_estimators_dict = {}

    # Run optimization for each model
    for acronym, create_model in models.items():
        print(f"Running Optuna optimization for {acronym}...\n")

        def objective(trial):
            model = create_model()
            param_ranges = tuner_param_ranges[acronym]

            params = {}

            for param_name, (low, high, param_type) in param_ranges.items():
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(param_name, low, high)
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(param_name, low, high)
                elif param_type == 'float_log':
                    params[param_name] = trial.suggest_float(param_name, low, high, log=True)

            model.set_params(**params)
            pipe = Pipeline([('model', model)])

            cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

            # Use 5-fold cross validation, random seeded to cv_splitter
            scores = cross_val_score(
                pipe,
                X,
                y,
                scoring='f1_weighted',  # Changed to classification metric
                cv=cv_splitter,
                n_jobs=n_jobs
            )

            return scores.mean()

        # Create and run study
        study = optuna.create_study(direction="maximize",
                                    study_name=f'{acronym}_sklearn',
                                    sampler=TPESampler(n_startup_trials=10))

        # Let's skip rfc's study
        # Takes too long, and it doesn't produce results comparable to the boosted tree algos
        if acronym == 'rfc':
            set_n_trials = 1
        else:
            set_n_trials = n_trials

        study.optimize(objective, n_trials=set_n_trials, n_jobs=n_jobs)

        # Get best model
        best_model = create_model()
        best_model.set_params(
            **study.best_params
        )

        pipe = Pipeline([('model', best_model)])
        pipe.fit(X, y)  # Fit on full dataset

        # Store best model results
        best_models_results.append({
            'model': acronym,
            'best_score': study.best_value,
            'best_params': study.best_params,
            'val_accuracy': cross_val_score(pipe, X, y, scoring='accuracy', cv=5).mean(),
            'val_f1_weighted': cross_val_score(pipe, X, y, scoring='f1_weighted', cv=5).mean()
        })

        # Store study results
        cv_results = pd.DataFrame([
            {
                'number': trial.number,
                'value': trial.value,
                **trial.params
            } for trial in study.trials
        ])

        export_df = cv_results.sort_values(by='value', ascending=False)
        export_df = export_df.head(5)  # You can get rid of this line to get all the results back

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            export_df.to_csv(
                os.path.join(output_dir, f"{acronym}_tuner_cv_results.csv"),
                index=False
            )

        # Store results
        cv_results_dict[acronym] = cv_results
        best_estimators_dict[acronym] = pipe

    # Create best models DataFrame
    best_models_df = pd.DataFrame(best_models_results).sort_values(
        'best_score', ascending=False
    )

    return {
        'best_models': best_models_df,
        'cv_results': cv_results_dict,
        'best_estimators': best_estimators_dict
    }


def skLearnPipeline(df, target_col, output_dir=None, random_seed=42, n_jobs=20, param_grids=None, cv=None):
    """
    Runs a grid search pipeline for multiple sklearn-compatible classification models.

    Parameters:
    -----------
    df : pandas DataFrame
        Preprocessed DataFrame ready for sklearn models
    target_col : str
        Name of the target column
    output_dir : str, optional
        Directory to save CV results. If None, results aren't saved to disk
    random_seed : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=20
        Number of jobs for parallel processing

    Returns:
    --------
    dict containing:
        - 'best_models': DataFrame with best models and their scores
        - 'cv_results': Dict of DataFrames with full CV results for each model
        - 'best_estimators': Dict of best fitted estimators for each model
    """

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()

    X_train_val, y_train_val, ps = get_train_val_ps(X_train, y_train, X_val, y_val)

    # Define models - converted from regressors to classifiers
    models = {
        'lr': LogisticRegression(random_state=random_seed, max_iter=1000),
        'sgdc': SGDClassifier(random_state=random_seed),
        'mlpc': MLPClassifier(
            early_stopping=True,
            random_state=random_seed,
            hidden_layer_sizes=(50, 50,)
        ),
        'rfc': RandomForestClassifier(random_state=random_seed),
        'xgbc': XGBClassifier(random_state=random_seed)
    }

    # Create pipelines
    pipes = {
        acronym: Pipeline([('model', model)])
        for acronym, model in models.items()
    }

    # Define parameter grids
    if not param_grids:
        param_grids = {
            'lr': [{
                'model__fit_intercept': [False, True],
                # 'model__C': [0.001, 0.01, 0.1, 1.0],
                # 'model__solver': ['liblinear', 'saga'],
                # 'model__penalty': ['l1', 'l2']
            }],

            'sgdc': [{
                'model__eta0': [0.1, 0.001, 0.00001],
                # 'model__alpha': [0.1, 0.001, 0.00001],
                # 'model__learning_rate': ['optimal', 'invscaling', 'adaptive'],
                # 'model__loss': ['log_loss', 'modified_huber', 'hinge']
            }],

            'mlpc': [{
                'model__alpha': [0.3, 0.1, 0.001],
                # 'model__learning_rate': ['constant', 'adaptive'],
                # 'model__activation': ['relu', 'tanh', 'logistic']
            }],

            'rfc': [{
                'model__n_estimators': [75, 150, 500],
                # 'model__max_depth': [None, 2, 5],
                # 'model__min_samples_split': [4, 7],
                # 'model__min_samples_leaf': [2, 3],
                # 'model__max_features': ['sqrt', 0.5],
                # 'model__class_weight': ['balanced', None]
            }],

            'xgbc': [{
                'model__max_depth': [None, 3, 7],
                # 'model__n_estimators': [75, 150, 300, 500],
                # 'model__learning_rate': [0.1, 0.001, 0.00001],
                # 'model__gamma': [0.5, 2, 3, 5]
            }]
        }

    # Store results
    best_models_results = []
    cv_results_dict = {}
    best_estimators_dict = {}

    # Run grid search for each model
    for acronym, pipe in pipes.items():
        print(f"Running GridSearchCV for {acronym}...\n")

        if acronym == 'xgbc':
            fit_params = {
                'model__eval_set': [(X_val, y_val)],
                'model__verbose': False
            }
        else:
            fit_params = {}

        if cv is None:
            warnings.warn(
                "Using predefined validation split may show optimistic performance "
                "compared to k-fold cross validation, as the dataset is small. "
                "Results may not be representative of real-world performance.",
                UserWarning
            )

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grids[acronym],
            scoring='f1_weighted',  # Changed from 'neg_mean_squared_error' to 'f1_weighted'
            n_jobs=n_jobs,
            cv=ps if (cv is None) else cv,
            return_train_score=True
        )

        # Fit
        gs.fit(X_train_val, y_train_val, **fit_params)

        # Get validation score from the predefined split results
        val_scores = gs.cv_results_['split0_test_score']
        best_val_score_idx = np.argmax(val_scores)
        # Using f1 score instead of r2 for classification
        val_f1 = cross_val_score(pipe.set_params(**gs.best_params_), X, y, scoring='f1_weighted', cv=5).mean()

        # Store best model results
        best_models_results.append({
            'model': acronym,
            'best_score': gs.best_score_,  # This is now accuracy from GridSearchCV
            'best_params': gs.best_params_,
            'val_f1_weighted': val_f1  # Using F1 score instead of R2
        })

        # Store CV results
        cv_results = pd.DataFrame(gs.cv_results_)
        important_columns = [
            'rank_test_score', 'mean_test_score', 'std_test_score',
            'mean_train_score', 'std_train_score', 'mean_fit_time',
            'std_fit_time', 'mean_score_time', 'std_score_time'
        ]

        # Reorder columns
        remaining_cols = [col for col in cv_results.columns if col not in important_columns]
        cv_results = cv_results[important_columns + remaining_cols]

        # Store results
        cv_results_dict[acronym] = cv_results
        best_estimators_dict[acronym] = gs.best_estimator_

        export_df = cv_results.sort_values(by='mean_test_score', ascending=False)
        export_df = export_df.head(5)

        # Save CV results if directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            export_df.to_csv(
                os.path.join(output_dir, f"{acronym}_gscv_results.csv"),
                index=False
            )

    # Create best models DataFrame
    best_models_df = pd.DataFrame(best_models_results).sort_values(
        'best_score', ascending=False
    )

    return {
        'best_models': best_models_df,
        'cv_results': cv_results_dict,
        'best_estimators': best_estimators_dict
    }


def skLearnOrchestrator(df, target_col, output_dir=None, random_seed=42, n_jobs=20, n_trials=1, cv=None,
                        param_grids=None, tuner_param_ranges=None, gridsearch=True, tuner=True):
    if (not gridsearch and not tuner):
        print(f'One of gridsearch or tuner must be set to true for this pipeline to run.')
        raise ValueError()

    all_best_models = []
    all_cv_results = {}
    all_best_estimators = {}

    gs_outputs = {
        'best_models': pd.DataFrame(),
        'cv_results': {},
        'best_estimators': {}
    }
    tuner_outputs = {
        'best_models': pd.DataFrame(),
        'cv_results': {},
        'best_estimators': {}
    }

    if gridsearch:
        gs_outputs = skLearnPipeline(df=df, target_col=target_col, output_dir=output_dir, random_seed=random_seed,
                                     n_jobs=n_jobs, cv=cv, param_grids=param_grids)
    if tuner:
        tuner_outputs = skLearnTuner(df=df, target_col=target_col, output_dir=output_dir, random_seed=random_seed,
                                     n_jobs=n_jobs, n_trials=n_trials, tuner_param_ranges=tuner_param_ranges)

    all_best_models = pd.concat([gs_outputs['best_models'],
                                 tuner_outputs['best_models']])
    all_cv_results.update(gs_outputs['cv_results'])
    all_cv_results.update(tuner_outputs['cv_results'])
    all_best_estimators.update(gs_outputs['best_estimators'])
    all_best_estimators.update(tuner_outputs['best_estimators'])

    return {
        'best_models': all_best_models,
        'cv_results': all_cv_results,
        'best_estimators': all_best_estimators
    }


class LightGBMOptunaCV(BaseEstimator):
    def __init__(self, scoring='f1_weighted', n_trials=1, n_jobs=2,
                 timeout=None, cv=5, random_state=42):
        """
        LightGBM optimizer using Optuna with cross-validation.

        Parameters:
        -----------
        scoring : str, default='accuracy'
            Scoring metric for optimization
            Options include 'accuracy', 'f1', 'auc', 'logloss'
        n_trials : int, default=100
            Number of Optuna study trials
        n_jobs : int, default=2
            Number of parallel jobs
            Threads for each model in lightgbm have been limited to 1
        timeout : int, optional
            Time limit in seconds for the optimization
        cv : int, default=5
            Number of cross-validation folds
        random_state : int, default=42
            Random state for cross-validation
        """
        self.scoring = scoring
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.cv = cv
        self.random_state = random_state
        self.study = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.best_iteration_ = None
        self.trials_dataframe = None
        self.n_parallel_trials = min(4, n_jobs)  # Move this up here

    def _objective(self, trial, X, y, categorical_features=None, param=None):
        # Begin with garbage collection
        gc.collect()

        if not param:
            param = {
                'objective': 'binary',  # Changed from 'regression' to 'binary'
                'verbosity': -1,
                'early_stopping_round': 50,
                'num_threads': 1,
                'feature_pre_filter': False,  # This was running into errors
                'use_missing': True,
                'zero_as_missing': False,
                'metric': 'binary_logloss',  # Added appropriate metric for binary classification

                # Some compute limiting for lightgbm, it uses too many threads
                'num_iterations_per_thread': 128,
                'force_row_wise': True,
                'max_threads_for_inner_ops': 1,
                'openmp_return_thread': 0,
                'num_parallel_tree': 1,
                # May avoid the issue of running into different hyperparameter configs
                'deterministic': True,

                # Parameters to optimize
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),
                # 100 to tighten the overfitting range on CV validation.
                'num_leaves': trial.suggest_int('num_leaves', 15, 200),  # Dropping this for now, just to avoid memory issues
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3, log=True),
                'learning_rate_decay': trial.suggest_float('learning_rate_decay', 0.8, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 2000),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),  # Needed to add this with pre_filter
                # Small dataset with a bunch of categoricals may benefit from this
                'min_data_per_group': trial.suggest_int('min_data_per_group', 10, 200),
                'max_cat_threshold': trial.suggest_int('max_cat_threshold', 16, 128),
                # 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.5),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.1, 1.0)
            }

        # LightGBM doesn't work with cross_val_score, so we need all this to stand in
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, valid_idx in kf.split(X):
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            # Create LightGBM datasets for this fold
            train_data = lgb.Dataset(
                X_train_fold,
                label=y_train_fold,
                categorical_feature=categorical_features,
                free_raw_data=True
            )
            valid_data = lgb.Dataset(
                X_valid_fold,
                label=y_valid_fold,
                categorical_feature=categorical_features,
                reference=train_data,
                free_raw_data=True
            )

            # Train model for this fold
            model = lgb.train(
                params=param,
                train_set=train_data,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            # Get validation score for this fold based on scoring metric
            y_pred = model.predict(X_valid_fold)

            if self.scoring == 'accuracy':
                y_pred_binary = (y_pred > 0.5).astype(int)
                fold_score = accuracy_score(y_valid_fold, y_pred_binary)
            elif self.scoring == 'f1':
                y_pred_binary = (y_pred > 0.5).astype(int)
                fold_score = f1_score(y_valid_fold, y_pred_binary)
            elif self.scoring == 'f1_weighted' or self.scoring == 'weighted':
                y_pred_binary = (y_pred > 0.5).astype(int)
                fold_score = f1_score(y_valid_fold, y_pred_binary, average='weighted')
            elif self.scoring == 'auc':
                fold_score = roc_auc_score(y_valid_fold, y_pred)
            elif self.scoring == 'logloss':
                fold_score = -log_loss(y_valid_fold, y_pred)  # Negative because Optuna maximizes
            else:
                # Default to accuracy if scoring not recognized
                y_pred_binary = (y_pred > 0.5).astype(int)
                fold_score = accuracy_score(y_valid_fold, y_pred_binary)

            scores.append(fold_score)
            gc.collect()

        return np.mean(scores)

    def fit(self, df, target_col, categorical_features=None):
        """
        Run Optuna optimization for LightGBM using cross-validation.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        target_col : str
            Name of target column
        categorical_features : list, optional
            List of categorical feature names
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Convert categorical features from names to indices if provided
        if categorical_features:
            categorical_indices = [X.columns.get_loc(col) for col in categorical_features]
        else:
            categorical_indices = None

        # Create Optuna study
        self.study = optuna.create_study(
            direction="maximize",
            study_name="lightgbm_optimization",
            sampler=TPESampler(n_startup_trials=2)
        )

        # Run optimization
        with parallel_backend('multiprocessing', n_jobs=self.n_parallel_trials):
            self.study.optimize(
                lambda trial: self._objective(trial, X, y, categorical_indices),
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
                timeout=self.timeout
            )

        # Store results
        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value
        self.trials_dataframe = self.study.trials_dataframe()

        # Train final model with best parameters
        best_params = self.best_params_.copy()
        best_params.update({
            'objective': 'binary',  # Changed from 'regression' to 'binary'
            'verbosity': -1,
            'early_stopping_round': 50,
            'num_threads': 1,
            'metric': 'binary_logloss'  # Added appropriate metric
        })

        train_data = lgb.Dataset(
            X,
            label=y,
            categorical_feature=categorical_indices,
            free_raw_data=True
        )

        self.best_estimator_ = lgb.train(
            params=best_params,
            train_set=train_data,
            valid_sets=[train_data]
        )

        self.best_iteration_ = self.best_estimator_.best_iteration

        return self

    def predict(self, X):
        """Make probability predictions with the best model."""
        if self.best_estimator_ is None:
            raise ValueError("Call fit before predict")
        return self.best_estimator_.predict(X, num_iteration=self.best_iteration_)

    def predict_proba(self, X):
        """Make probability predictions with the best model."""
        if self.best_estimator_ is None:
            raise ValueError("Call fit before predict_proba")
        probas = self.best_estimator_.predict(X, num_iteration=self.best_iteration_)
        # Format as 2D array with probabilities for both classes
        return np.vstack([1 - probas, probas]).T

    def predict_classes(self, X, threshold=0.5):
        """Make class predictions with the best model."""
        if self.best_estimator_ is None:
            raise ValueError("Call fit before predict_classes")
        probas = self.best_estimator_.predict(X, num_iteration=self.best_iteration_)
        return (probas > threshold).astype(int)


def lightgbmTuner(df, target_col, categorical_features=None,
                  output_dir=None, n_jobs=2, n_trials=1, timeout=None,
                  cv=5, random_state=42):
    """
    Runs Optuna optimization for LightGBM classification model with compatible output format to skLearnPipeline.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    target_col : str
        Name of target column
    categorical_features : list, optional
        List of categorical feature names
    output_dir : str, optional
        Directory to save optimization results
    n_jobs : int, default=2
        Number of parallel jobs
    n_trials : int, default=100
        Number of optimization trials
    timeout : int, optional
        Time limit in seconds for the optimization
    cv : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random state for cross-validation

    Returns:
    --------
    dict containing:
        - 'best_models': DataFrame with best model and scores
        - 'cv_results': Dict of DataFrames with optimization results
        - 'best_estimators': Dict with best fitted estimator
    """

    print("Running Optuna optimization for LightGBM Classification...\n")

    # Perform optimization
    opt = LightGBMOptunaCV(
        scoring='f1_weighted',  # Changed from MSE to f1_weighted, in line with the other pipelines
        n_trials=n_trials,
        n_jobs=n_jobs,
        timeout=timeout,
        cv=cv,
        random_state=random_state
    )
    # Fit model
    opt.fit(df, target_col, categorical_features)

    # Calculate validation metrics
    X = df.drop(columns=[target_col])
    y = df[target_col]
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_predictions = []
    cv_actuals = []

    for train_idx, valid_idx in cv_splitter.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        params = opt.best_params_.copy()
        params.update({'objective': 'binary', 'metric': 'binary_logloss'})
        model = lgb.train(params, train_data)
        y_pred = model.predict(X_valid)
        y_pred_binary = (y_pred > 0.5).astype(int)

        cv_predictions.extend(y_pred_binary)
        cv_actuals.extend(y_valid)

    # Calculate classification metrics
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    # Convert predictions to numpy arrays
    cv_predictions = np.array(cv_predictions)
    cv_actuals = np.array(cv_actuals)

    val_accuracy = accuracy_score(cv_actuals, cv_predictions)
    val_f1 = f1_score(cv_actuals, cv_predictions, average='weighted')

    # Create results in same format as skLearnPipeline
    best_models_df = pd.DataFrame([{
        'model': 'lgbm',
        'best_score': opt.best_score_,
        'best_params': opt.best_params_,
        'val_accuracy': val_accuracy,
        'val_f1_weighted': val_f1
    }])

    export_df = opt.trials_dataframe.sort_values(by='value', ascending=False)
    export_df = export_df.head(5)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        export_df.to_csv(
            os.path.join(output_dir, 'lgbm_optuna_results.csv'),
            index=False
        )

    return {
        'best_models': best_models_df,
        'cv_results': {'lgbm': opt.trials_dataframe},
        'best_estimators': {'lgbm': opt.best_estimator_}
    }


def lightgbmOrchestrator():
    pass


def catboostPipe():
    # Takes in string data as part of the categorical processing, and nulls, get -mse
    # Wait to implement
    pass


def catboostTuner(df, target_col, categorical_features=None,
                  output_dir=None, n_jobs=3, n_trials=100, timeout=None,
                  cv=5, random_state=42):
    pass
    # """
    # Tune CatBoost hyperparameters using Optuna.
    #
    # Args:
    #     df: pandas DataFrame containing features and target
    #     target_col: string, name of the target column
    #     categorical_features: list of categorical column names
    #     output_dir: string, directory to save model artifacts
    #     n_jobs: int, number of parallel jobs
    #     n_trials: int, number of optimization trials
    #     timeout: int, timeout in seconds for optimization
    #     cv: int, number of cross-validation folds
    #     random_state: int, random state for reproducibility
    #
    # Returns:
    #     dict with best parameters and best score
    # """
    #
    # # Prepare data
    # X = df.drop(columns=[target_col])
    # y = df[target_col]
    #
    # splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    #
    # # Initialize lists to store trial information
    # trials_list = []
    #
    # def objective(trial):
    #     # Common parameters
    #     param = {
    #         'iterations': trial.suggest_int('iterations', 100, 1000),
    #         'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
    #         'depth': trial.suggest_int('depth', 4, 10),
    #         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
    #         "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
    #         'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
    #         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
    #         'od_wait': trial.suggest_int('od_wait', 10, 50),
    #         'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
    #         "od_type": "Iter",
    #         'random_state': random_state,
    #         'verbose': False
    #     }
    #
    #     # Handle different bootstrap types
    #     if param["bootstrap_type"] == "Bernoulli":
    #         param["subsample"] = trial.suggest_float("subsample", 0.1, 1.0)
    #
    #     # Initialize scores list
    #     fold_scores = []
    #
    #     # Perform cross-validation with Pool objects
    #     for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
    #         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    #         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    #
    #         # Create Pool objects
    #         train_pool = Pool(
    #             data=X_train,
    #             label=y_train,
    #             cat_features=categorical_features
    #         )
    #         val_pool = Pool(
    #             data=X_val,
    #             label=y_val,
    #             cat_features=categorical_features
    #         )
    #
    #         # Initialize and train model
    #         model = CatBoostRegressor(**param)
    #         eval_metric = 'RMSE'
    #
    #         # Fit model
    #         model.fit(
    #             train_pool,
    #             eval_set=val_pool,
    #             use_best_model=True,
    #             early_stopping_rounds=50,
    #             verbose=False
    #         )
    #
    #         # Get validation score
    #         score = model.get_best_score()['validation'][eval_metric]
    #         score = -score  # Negative for minimization
    #         fold_scores.append(score)
    #
    #         trials_list.append({
    #             'number': trial.number,
    #             'value': score,
    #             'params': trial.params,
    #             'state': trial.state,
    #         })
    #
    #     # Return mean score
    #     return np.mean(fold_scores)
    #
    # # Create study
    # study = optuna.create_study(direction='minimize')
    #
    # # Optimize
    # study.optimize(
    #     objective,
    #     n_trials=n_trials,
    #     timeout=timeout,
    #     # Setting n_jobs here, CatBoost (like LightGBM) has internal parallelization so be careful running more than 2
    #     n_jobs=n_jobs
    # )
    #
    # trials_df = pd.DataFrame(trials_list)
    #
    # # Get best parameters
    # best_params = study.best_params
    # best_score = -study.best_value  # Convert back to positive score
    #
    # # Save study if output directory is provided
    # if output_dir:
    #     os.makedirs(output_dir, exist_ok=True)
    #     study.trials_dataframe().to_csv(
    #         os.path.join(output_dir, 'placeholder.csv'),
    #         index=False
    #     )
    #
    # # Train final model with best parameters
    # # Create final Pool object with all data
    # final_pool = Pool(
    #     data=X,
    #     label=y,
    #     cat_features=categorical_features
    # )
    #
    # final_model = CatBoostRegressor(**best_params)
    #
    # best_models_df = pd.DataFrame([{
    #     'model': 'catboost',
    #     'best_score': study.best_value,
    #     'best_params': best_params,
    #     # 'val_r2': val_r2
    # }])
    #
    # return {
    #     'best_models': best_models_df,
    #     'cv_results': {'cat': trials_df},
    #     'best_estimators': {'cat': final_model}
    # }


def combine_pipeline_results(sklearn_results, lightgbm_results, catboost_results=None):
    # Combine best_models DataFrames
    combined_best_models = pd.concat([
        sklearn_results['best_models'],
        lightgbm_results['best_models']
    ], axis=0).reset_index(drop=True)

    # Sort by best_score to maintain ranking
    combined_best_models = combined_best_models.sort_values(
        'best_score', ascending=False
    ).reset_index(drop=True)

    # Merge cv_results dictionaries
    combined_cv_results = {
        **sklearn_results['cv_results'],
        **lightgbm_results['cv_results']
    }

    # Merge best_estimators dictionaries
    combined_best_estimators = {
        **sklearn_results['best_estimators'],
        **lightgbm_results['best_estimators']
    }

    return {
        'best_models': combined_best_models,
        'cv_results': combined_cv_results,
        'best_estimators': combined_best_estimators
    }


def pipeline_predict(pipeline_results, test_df, target_col=None, id_columns=None):
    """
    Make predictions using the best models from the classification pipeline.

    Parameters:
    -----------
    pipeline_results : dict
        Results dictionary from skLearnPipeline containing 'best_estimators'
    test_df : pandas DataFrame
        Test data to make predictions on
    target_col : str, optional
        Name of target column if present in test_df
        For inference and testing the inference

    Returns:
    --------
    dict containing:
        - 'predictions': DataFrame with predictions from all models
        - 'metrics': DataFrame with test metrics (if target_col provided)
    """

    # Prepare test features
    drop_cols = [col for col in (id_columns + [target_col] if target_col else id_columns)
                 if col in test_df.columns]
    X_test = test_df.drop(columns=drop_cols)

    # Initialize results dictionary
    results = {'predictions': {}, 'probabilities': {}}

    # Get predictions from each model
    for model_name, estimator in pipeline_results['best_estimators'].items():
        # Get class predictions
        predictions = pd.Series(
            estimator.predict(X_test) if not isinstance(estimator, lgb.Booster)
            else (estimator.predict(X_test) > 0.5).astype(int),
            index=test_df.index,
            name=model_name
        )
        results['predictions'][model_name] = predictions

        # Check if the model supports probability estimates
        if hasattr(estimator, 'predict_proba'):
            # Store probability predictions if available
            proba = estimator.predict_proba(X_test)
            # For binary classification, store probability of positive class
            if proba.shape[1] == 2:
                results['probabilities'][f"{model_name}_proba"] = pd.Series(
                    proba[:, 1],
                    index=test_df.index,
                    name=f"{model_name}_proba"
                )
            else:
                # For multi-class, store all probabilities
                for i, class_name in enumerate(estimator.classes_):
                    results['probabilities'][f"{model_name}_proba_class_{class_name}"] = pd.Series(
                        proba[:, i],
                        index=test_df.index,
                        name=f"{model_name}_proba_class_{class_name}"
                    )
        elif isinstance(estimator, lgb.Booster):
            # Handle LightGBM booster directly - its predict() returns probabilities
            proba = estimator.predict(X_test)
            # For binary classification in LightGBM, add to probabilities dict
            results['probabilities'][f"{model_name}_proba"] = pd.Series(
                proba,
                index=test_df.index,
                name=f"{model_name}_proba"
            )

    # Convert predictions to DataFrame
    results['predictions'] = pd.DataFrame(results['predictions'])

    if results['probabilities']:
        results['probabilities'] = pd.DataFrame(results['probabilities'])

    # Check index after conversion
    assert test_df.index.equals(results['predictions'].index)

    # Join predictions with original test data
    output_df = pd.concat([test_df, results['predictions']], axis=1, verify_integrity=True)

    # Add probability predictions if available
    if results['probabilities'] is not None and not results['probabilities'].empty:
        output_df = pd.concat([output_df, results['probabilities']], axis=1, verify_integrity=True)

    # Calculate metrics if target is provided
    if target_col in test_df.columns:
        y_test = test_df[target_col]
        metrics = []

        for model_name in results['predictions'].columns:
            y_pred = results['predictions'][model_name]

            # Calculate classification metrics
            metrics.append({
                'model': model_name,
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'test_recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'test_f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            })

            # Add confusion matrix
            results[f'confusion_matrix_{model_name}'] = confusion_matrix(y_test, y_pred)

        results['metrics'] = pd.DataFrame(metrics)

    return output_df, results


def save_feature_importance(pipeline_results, X, output_dir):
    """
    Extract and save feature importance from tree-based models.

    Parameters:
    -----------
    pipeline_results : dict
        Dictionary containing best_estimators from the pipeline
    X : pandas DataFrame
        DataFrame with feature names
    output_dir : str
        Directory to save feature importance results
    """
    feature_importance_dfs = []

    # Get feature names
    feature_names = X.columns.tolist()

    # Track seen models to handle duplicates
    seen_models = {}

    for model_name, pipeline in pipeline_results['best_estimators'].items():
        # Check if the model is a tree-based model that supports feature importance
        if model_name.lower() in ['rfc', 'xgbc', 'lgbm']:
            try:
                # Extract the model from the pipeline
                if hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
                    model = pipeline.named_steps['model']
                else:
                    model = pipeline  # In case it's not a pipeline

                # Get feature importance based on model type
                if model_name.lower() == 'rfc':
                    importance = model.feature_importances_
                elif model_name.lower() == 'xgbc':
                    importance = model.feature_importances_
                elif model_name.lower() == 'lgbm':
                    # Handle both scikit-learn wrapper and native LightGBM Booster
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    elif hasattr(model, 'booster_'):
                        importance = model.booster_.feature_importance(importance_type='gain')
                    # Handle pure LightGBM Booster object
                    elif hasattr(model, 'feature_importance'):
                        print('This is a native Booster object, drop the original two ifs/elifs')
                        importance = model.feature_importance(importance_type='gain')
                    else:
                        print(f"Cannot extract feature importance from {model_name} (type: {type(model)})")
                        continue
                else:
                    continue  # Skip if not a supported model

                # Create DataFrame for this model's feature importance
                imp_df = pd.DataFrame({
                    'feature': feature_names[:len(importance)],  # Ensure lengths match
                    'importance': importance
                })

                # Sort by importance
                imp_df = imp_df.sort_values('importance', ascending=False)

                # Create a unique file name for the model
                if model_name in seen_models:
                    seen_models[model_name] += 1
                    unique_name = f"{model_name}_tuner_{seen_models[model_name]}"
                else:
                    seen_models[model_name] = 0
                    unique_name = f"{model_name}_grid"

                # Add model name with type indicator
                imp_df['model'] = unique_name

                # Add to collection
                feature_importance_dfs.append(imp_df)

                # Save individual model feature importance
                if output_dir:
                    imp_df.to_csv(
                        os.path.join(output_dir, f"{unique_name}_feature_importance.csv"),
                        index=False
                    )

                    # Create a simple bar plot of feature importance
                    plt.figure(figsize=(10, 8))
                    top_features = imp_df.head(20)  # Show top 20 features
                    sns.barplot(x='importance', y='feature', data=top_features)
                    plt.title(f'Top 20 Feature Importance - {unique_name}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{unique_name}_feature_importance.png"))
                    plt.close()

                print(f"Successfully extracted feature importance for {model_name}")

            except Exception as e:
                print(f"Error extracting feature importance for {model_name}: {e}")
                import traceback
                traceback.print_exc()  # Print full traceback for debugging

    # Combine all feature importance DataFrames if there are any
    if feature_importance_dfs:
        all_importance = pd.concat(feature_importance_dfs, axis=0)

        if output_dir:
            all_importance.to_csv(
                os.path.join(output_dir, "All_Feature_Importance.csv"),
                index=False
            )

        return all_importance
    else:
        print('No feature importance to join.')
        return None


def shallowMachineLearningPipeline(df, test_df, ordinal_columns, categorical_columns, numerical_columns, target_col,
                                   id_columns=None, datetime_columns=None, output_dir=None):

    # Convert id_columns to list if string
    if isinstance(id_columns, str):
        id_columns = [id_columns]

    if isinstance(datetime_columns, str):
        datetime_columns = [datetime_columns]


    # # Print target column before preprocessing
    # # Prepre analytics
    # print(f"\n{'=' * 50}\nTARGET COLUMN BEFORE PREPROCESSING\n{'=' * 50}")
    # print(f"Unique values in {target_col}: {df[target_col].unique()}")
    # print(f"Data type of {target_col}: {df[target_col].dtype}")
    # print(f"Value counts of {target_col}:\n{df[target_col].value_counts(dropna=False)}")
    #
    # # Print categorical columns before preprocessing
    # print(f"\n{'=' * 50}\nCATEGORICAL COLUMNS BEFORE PREPROCESSING\n{'=' * 50}")
    # for col in categorical_columns:
    #     print(f"Unique values in {col}: {df[col].unique()}")
    #     print(f"Data type of {col}: {df[col].dtype}")
    #     print(f"Value counts of {col}:\n{df[col].value_counts(dropna=False)}\n")


    data_dict, categorical_columns = dataPreprocessor(df, ordinal_columns, categorical_columns, numerical_columns,
                                                      target_col, id_columns=id_columns,
                                                      datetime_columns=datetime_columns, test_df=test_df)


    # # Print target column after preprocessing
    # # Post datapreprocessing
    # print(f"\n{'=' * 50}\nTARGET COLUMN AFTER PREPROCESSING\n{'=' * 50}")
    # for df_name, df_dict in data_dict.items():
    #     if df_name != 'fitted_encoders':
    #         for subset_name, df in df_dict.items():
    #             if target_col in df.columns:
    #                 print(f"Dataset: {df_name}, Subset: {subset_name}")
    #                 print(f"Unique values in {target_col}: {df[target_col].unique()}")
    #                 print(f"Data type of {target_col}: {df[target_col].dtype}")
    #                 print(f"Value counts of {target_col}:\n{df[target_col].value_counts(dropna=False)}\n")
    #
    # # Print categorical columns after preprocessing
    # print(f"\n{'=' * 50}\nCATEGORICAL COLUMNS AFTER PREPROCESSING\n{'=' * 50}")
    # for df_name, df_dict in data_dict.items():
    #     if df_name != 'fitted_encoders':
    #         for subset_name, df in df_dict.items():
    #             print(f"Dataset: {df_name}, Subset: {subset_name}")
    #             for col in categorical_columns:
    #                 if col in df.columns:
    #                     print(f"Unique values in {col}: {df[col].unique()}")
    #                     print(f"Data type of {col}: {df[col].dtype}")
    #                     print(f"Value counts of {col}:\n{df[col].value_counts(dropna=False)}\n")


    # Save the fitted encoders
    joblib.dump(data_dict['fitted_encoders'], rf'{output_dir}/fitted_encoders.joblib')

    print(f'New Categorical Columns after preprocessing: {categorical_columns}\n')

    # Print dataset info
    for df_name, df_dict in data_dict.items():
        if df_name != 'fitted_encoders':
            print(f"\n{'=' * 50}\n{df_name.upper()}\n{'=' * 50}")
            for subset_name, df in df_dict.items():
                print(f"\n----- {subset_name.upper()} DATASET -----")
                print("\nHead:")
                print(df.head())
                print("\nInfo:")
                print(df.info())
                print("\nColumns:")
                print(df.columns)

    data_dict['sklearn_ready']['train'].drop(columns=id_columns, inplace=True)
    data_dict['lightgbm_ready']['train'].drop(columns=id_columns, inplace=True)
    data_dict['catboost_ready']['train'].drop(columns=id_columns, inplace=True)

    # Use training data for model fitting
    # Skip for testing lightgbm's portion
    sklearn_results = skLearnOrchestrator(df=data_dict['sklearn_ready']['train'], target_col=target_col,
                                   output_dir=output_dir, cv=5)
    print("\nBest Models Summary:")
    print(f"\n{sklearn_results['best_models']}")
    print("\nRandom Forest CV Results:")
    print(f"\n{sklearn_results['cv_results']['rfc']}")
    print("\nXGBoosted Trees CV Results:")
    print(f"\n{sklearn_results['cv_results']['xgbc']}")

    # Use training data for LightGBM
    lgb_df = data_dict['lightgbm_ready']['train']

    # Split into features and target
    X = lgb_df.drop(columns=[target_col])
    y = lgb_df[target_col]

    # Get categorical features (just both ordinal+categorical since lightgbm can handle both)
    categorical_features = categorical_columns + ordinal_columns


    lightgbm_results = lightgbmTuner(df=data_dict['lightgbm_ready']['train'],
                                    target_col=target_col,
                                    categorical_features=categorical_features,
                                    cv=5,
                                    random_state=42, output_dir=output_dir)

    print("\nBest Models Summary:")
    print(f"{lightgbm_results['best_models']}\n")
    print("\nCross-validation Results:")
    cv_results = lightgbm_results['cv_results']['lgbm']
    print(f"{pd.DataFrame(cv_results).filter(like='mean_test').head()}\n")
    print("\nBest Model Parameters:")
    print(f"{lightgbm_results['best_estimators']['lgbm']}\n")

    # Combine results and make predictions
    pipeline_results = combine_pipeline_results(sklearn_results, lightgbm_results)

    # But before predictions, save the feature importances
    X_features = data_dict['sklearn_ready']['train'].drop(columns=[target_col])
    feature_importance = save_feature_importance(pipeline_results, X_features, output_dir)

    if output_dir:
        pipeline_results['best_models'].to_csv(
            os.path.join(output_dir, f"Best_Models.csv"),
            index=False
        )

        # Initialize an empty set
        seen_models = set()

        for i, (model_name, pipeline) in enumerate(pipeline_results['best_estimators'].items()):
            # GridSearchModels go first, any duplicates are from the tuner.
            print(f'Iteration {i} is model: {model_name}')

            if model_name in seen_models:
                model_path = os.path.join(output_dir, f'{model_name}_pipeline_tuner')
            else:
                model_path = os.path.join(output_dir, f'{model_name}_pipeline')
                seen_models.add(model_name)

            print(f'Created model_path: {model_path}, will be saved here.')

            # Save the model
            joblib.dump(pipeline, f"{model_path}.joblib")
            print(f"Saved model {model_name} to {model_path}.joblib")


            # Original code that still needs to be fixed below
            # Originally, I wanted to create their native files so the XGBR file could be
            # used across different applications, but we're saving the fitted_encoders in a joblib python native file
            # so that entire section needs to be redone anyways, for cross-application use
            # named_steps is the incorrect pipeline call though.
            # DONT USE BELOW, DOES NOT WORK


            # print(f'{model_name}')
            # model_path = os.path.join(output_dir, f'{model_name}_pipeline')
            #
            # if 'cat' == model_name.lower():
            #     model = pipeline.named_steps[f'{model_name}']
            #     model.save_model(f"{model_path}.cbm")
            # elif 'xgbc' == model_name.lower():
            #     model = pipeline.named_steps[f'{model_name}']
            #     model.save_model(f"{model_path}.model")
            # elif 'lgbm' == model_name.lower():
            #     model = pipeline.named_steps[f'{model_name}']
            #     model.booster_.save_model(f"{model_path}.txt")
            # else:
            #     joblib.dump(pipeline, f"{model_path}.joblib")


    # Make predictions using appropriate preprocessed test data
    sk_output_df, sklearn_predictions = pipeline_predict(
        {'best_estimators': {k: v for k, v in pipeline_results['best_estimators'].items()
                             if k in sklearn_results['best_estimators']}},
        data_dict['sklearn_ready']['test'],
        target_col, id_columns=id_columns
    )

    lgbm_output_df, lightgbm_predictions = pipeline_predict(
        {'best_estimators': {k: v for k, v in pipeline_results['best_estimators'].items()
                             if k in lightgbm_results['best_estimators']}},
        data_dict['lightgbm_ready']['test'],
        target_col, id_columns=id_columns
    )

    # Combine predictions
    all_predictions = {
        'predictions': pd.concat([
            sklearn_predictions['predictions'],
            lightgbm_predictions['predictions']
        ], axis=1)
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        all_predictions['predictions'].to_csv(
            os.path.join(output_dir, f"All_Predictions.csv"),
            index=False
        )
        if not sk_output_df.empty:
            sk_output_df.to_csv(
                os.path.join(output_dir, f"Sklearn_Outputs.csv"),
                index=False
            )
        if not lgbm_output_df.empty:
            lgbm_output_df.to_csv(
                os.path.join(output_dir, f"Lgbm_Outputs.csv"),
                index=False
            )

    # Combine metrics if they exist
    if 'metrics' in sklearn_predictions and 'metrics' in lightgbm_predictions:
        all_predictions['metrics'] = pd.concat([
            sklearn_predictions['metrics'],
            lightgbm_predictions['metrics']
        ], axis=0).reset_index(drop=True)

        if output_dir:
            all_predictions['metrics'].to_csv(
                os.path.join(output_dir, f"All_Metrics.csv"),
                index=False
            )

    # stats.model linear model analysis, just to make sure everything lines up currently
    # Gives very close results to my Scikitlearn, but validation is still going crazy.
    # Maybe because of how I'm calculating it?
    # For now, I'll ignore it and run it out some more 'test-run' datasets, at least the model is fitting correctly.

    # y_train = data_dict['sklearn_ready']['train'][target_col]
    # X_train = data_dict['sklearn_ready']['train'].drop(columns=target_col)
    #
    # y_test = data_dict['sklearn_ready']['test'][target_col]
    # X_test = data_dict['sklearn_ready']['test'].drop(columns=target_col)
    # X_test = X_test.drop(columns=id_columns)
    #
    # ols_results = sm.OLS(y_train, X_train, fit_intercept=False).fit()
    # print(ols_results.summary())
    #
    # y_pred = ols_results.predict(X_test)
    #
    # # Your metrics verification remains the same
    # r2 = r2_score(y_test, y_pred)
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # mae = mean_absolute_error(y_test, y_pred)
    #
    # print("\nTest Set Performance Metrics:")
    # print(f"R² Score:     {r2:.4f}")
    # print(f"RMSE:         {rmse:.4f}")
    # print(f"MAE:          {mae:.4f}")

    return all_predictions, pipeline_results, feature_importance


def main():
    pass


if __name__ == '__main__':
    main()