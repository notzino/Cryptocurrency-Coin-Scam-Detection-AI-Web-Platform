import warnings
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from xgboost import XGBClassifier, DMatrix, train
from sklearn.metrics import classification_report, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import numpy as np
from data_processing.data_loader import load_data_from_db
from extract_gpt_features import extract_features_for_model
from data_processing.preprocessing import clean_historical_data, clean_news_data, clean_social_media_data
import shap
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings('default', category=UserWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='xgboost')

LIMITED_DATA_THRESHOLD = 3
CRYPTO_KEYWORDS = ["cryptocurrency", "crypto", "blockchain", "coin", "token", "ICO", "exchange", "wallet"]


def ensure_two_class_proba(y_prob):
    if y_prob.ndim == 1 or y_prob.shape[1] == 1:
        y_prob = np.hstack([1 - y_prob.reshape(-1, 1), y_prob.reshape(-1, 1)])
    return y_prob


def custom_roc_auc_score(y_true, y_prob):
    # Ensure two-class probabilities
    y_prob = ensure_two_class_proba(y_prob)
    if len(np.unique(y_true)) == 1:
        return 0.5
    return roc_auc_score(y_true, y_prob[:, 1])


roc_auc_scorer = make_scorer(custom_roc_auc_score)


def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def add_temporal_features(df, time_column):
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df.reset_index(drop=True, inplace=True)
    return df


def add_aggregation_features(df, group_column):
    df_grouped = df.groupby(group_column)
    df_agg = df_grouped.agg(['mean', 'std', 'min', 'max'])
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    df = df.merge(df_agg, on=group_column, how='left')
    return df


def preprocess_and_extract_features(X, y):
    correlation_matrix = pd.DataFrame(X).corr().abs()
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_reduced = pd.DataFrame(X).drop(columns=to_drop).values

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_reduced)

    constant_filter = VarianceThreshold(threshold=0.0)
    X_constant = constant_filter.fit_transform(X_imputed)

    num_features = min(X_constant.shape[1], 10)
    select_k_best = SelectKBest(mutual_info_classif, k=num_features)
    X_selected = select_k_best.fit_transform(X_constant, y)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_selected)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_poly)

    return X_pca, pca, constant_filter, select_k_best, to_drop, poly, imputer


def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0.1, 1, 10, 50],
        'reg_lambda': [0.1, 1, 10, 50]
    }

    model = XGBClassifier(objective='binary:logistic', random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_


def select_features_with_shap(model, X_train, y_train):
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)

    shap.summary_plot(shap_values, X_train)

    feature_importance = np.abs(shap_values.values).mean(axis=0)
    important_features = np.argsort(feature_importance)[-10:]

    return important_features


def filter_relevant_info(df, content_column):
    keyword_pattern = '|'.join(CRYPTO_KEYWORDS)
    return df[df[content_column].str.contains(keyword_pattern, case=False, na=False)]


def extract_additional_features(historical_df, wallet_data_df):
    if 'Volume' in historical_df.columns:
        historical_df['avg_volume'] = historical_df['Volume'].rolling(window=30).mean()
        historical_df['volatility'] = historical_df['Close'].rolling(window=30).std()
    else:
        historical_df['avg_volume'] = np.nan
        historical_df['volatility'] = np.nan

    if 'balance_change' in wallet_data_df.columns:
        wallet_data_df['balance_change_avg'] = wallet_data_df['balance_change'].rolling(window=30).mean()
    else:
        wallet_data_df['balance_change_avg'] = np.nan

    return historical_df, wallet_data_df

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring="roc_auc", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training Score")
    plt.plot(train_sizes, test_scores_mean, label="Cross-Validation Score")
    plt.xlabel("Training Set Size")
    plt.ylabel("ROC-AUC Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def train_supervised_model(features_df, coins_df, retrain=False):
    if 'coin_id' not in features_df.columns:
        raise ValueError("features_df must contain 'coin_id' column.")
    if 'id' not in coins_df.columns or 'is_scam' not in coins_df.columns:
        raise ValueError("coins_df must contain 'id' and 'is_scam' columns.")

    features_df = features_df.drop_duplicates()

    X = features_df.drop(columns=['coin_id'])
    y = coins_df.set_index('id').loc[features_df['coin_id']]['is_scam']

    if len(np.unique(y)) != 2:
        raise ValueError("Target variable 'y' does not have exactly two unique values.")

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    legit_indices = y[y == 0].index[:16]
    scam_indices = y[y == 1].index[:15]

    if len(legit_indices) < 10 or len(scam_indices) < 10:
        raise ValueError("Not enough samples to select 10 of each class.")

    selected_indices = np.concatenate([legit_indices, scam_indices])

    if not set(selected_indices).issubset(X.index):
        raise ValueError("Selected indices do not match DataFrame indices.")

    X = X.loc[selected_indices]
    y = y.loc[selected_indices]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    undersample = RandomUnderSampler(random_state=42)
    smote = SMOTE(random_state=42)
    X_res, y_res = undersample.fit_resample(X, y)
    X_res, y_res = smote.fit_resample(X_res, y_res)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    X_train_pca, pca, constant_filter, select_k_best, to_drop, poly, imputer = preprocess_and_extract_features(X_train,
                                                                                                               y_train)
    X_test_reduced = pd.DataFrame(X_test).drop(columns=to_drop).values
    X_test_imputed = imputer.transform(X_test_reduced)
    X_test_poly = poly.transform(select_k_best.transform(constant_filter.transform(X_test_imputed)))
    X_test_pca = pca.transform(X_test_poly)

    best_params = hyperparameter_tuning(X_train_pca, y_train)
    model = XGBClassifier(**best_params)

    important_features = select_features_with_shap(model, X_train_pca, y_train)
    X_train_selected = X_train_pca[:, important_features]
    X_test_selected = X_test_pca[:, important_features]

    dtrain = DMatrix(X_train_selected, label=y_train)
    dtest = DMatrix(X_test_selected, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'random_state': 42,
        'max_depth': best_params['max_depth'],
        'learning_rate': best_params['learning_rate'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'scale_pos_weight': 1,
        'reg_alpha': best_params['reg_alpha'],
        'reg_lambda': best_params['reg_lambda'],
        'eval_metric': 'auc',
        'verbosity': 1
    }

    evals = [(dtrain, 'train'), (dtest, 'eval')]

    print("Starting model training with early stopping...")
    model_xgb = train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=10)
    print("Model training completed.")

    y_pred = (model_xgb.predict(dtest) > 0.5).astype(int)
    y_prob = model_xgb.predict(dtest)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob)}")

    joblib.dump({'model': model_xgb, 'scaler': scaler, 'constant_filter': constant_filter, 'pca': pca,
                 'select_k_best': select_k_best, 'to_drop': to_drop, 'poly': poly, 'imputer': imputer},
                'trained_model.pkl')

    plot_learning_curve(model, X_train_selected, y_train)

    # Plot the ROC curve
    plot_roc_curve(y_test, y_prob)

    return model_xgb


def main(retrain=False):
    coins_df, historical_df, news_df, social_media_df, wallet_data_df, transactions_df = load_data_from_db()
    historical_df = clean_historical_data(historical_df)
    news_df = clean_news_data(news_df)
    social_media_df = clean_social_media_data(social_media_df)

    news_df = filter_relevant_info(news_df, 'description')
    social_media_df = filter_relevant_info(social_media_df, 'content')

    historical_df, wallet_data_df = extract_additional_features(historical_df, wallet_data_df)

    features_df = extract_features_for_model(coins_df, historical_df, news_df, social_media_df, wallet_data_df,
                                             transactions_df)

    features_df = features_df.drop_duplicates()

    coins_with_insufficient_data = features_df[features_df['coin_id'].isin(
        social_media_df['coin_id'].value_counts()[
            social_media_df['coin_id'].value_counts() < LIMITED_DATA_THRESHOLD].index
    ) | features_df['coin_id'].isin(
        news_df['coin_id'].value_counts()[news_df['coin_id'].value_counts() < LIMITED_DATA_THRESHOLD].index
    ) | features_df['coin_id'].isin(
        historical_df['coin_id'].value_counts()[historical_df['coin_id'].value_counts() < LIMITED_DATA_THRESHOLD].index
    )]['coin_id'].unique()

    coins_df.loc[coins_df['id'].isin(coins_with_insufficient_data), 'is_scam'] = 1

    coins_df['is_scam'] = coins_df['is_scam'].astype('int64')

    model = train_supervised_model(features_df, coins_df, retrain=retrain)
    return model


if __name__ == '__main__':
    retrain = True
    model = main(retrain=retrain)
