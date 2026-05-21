import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ==============================================================
# 1. LOAD DATA
# ==============================================================

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'level'
]

print("Loading data...")
df_train = pd.read_csv(train_url, names=columns)
df_test = pd.read_csv(test_url, names=columns)

# Drop difficulty level column (not a feature)
df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'], inplace=True)

print(f"Training set: {df_train.shape[0]} records, {df_train.shape[1]} columns")
print(f"Test set:     {df_test.shape[0]} records, {df_test.shape[1]} columns")

# ==============================================================
# 2. DEFINE CATEGORICAL FEATURES
# ==============================================================

# Merge temporarily to ensure consistent encoding across train and test
df_full = pd.concat([df_train, df_test])

cat_cols = ['protocol_type', 'service', 'flag']

# ==============================================================
# 3. MAP ATTACKS TO 5 CATEGORIES
# ==============================================================

category_map = {
    'normal': 'Normal',
    # DoS
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    # Probe
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'ftp_write': 'R2L',
    'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L', 'warezmaster': 'R2L',
    'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')

# ==============================================================
# 4. PREPARE FEATURES AND LABELS
# ==============================================================

# Drop constant column and original class labels
df_full.drop(columns=['num_outbound_cmds', 'class'], inplace=True)

# Split back into train and test
train_len = len(df_train)
df_train_processed = df_full.iloc[:train_len].copy()
df_test_processed = df_full.iloc[train_len:].copy()

X_train = df_train_processed.drop(columns=['category'])
y_train = df_train_processed['category']

X_test = df_test_processed.drop(columns=['category'])
y_test = df_test_processed['category']

print(f"\nFeatures: {X_train.shape[1]}")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())

# ==============================================================
# YOUR WORK STARTS HERE
# ==============================================================

categorical_features = cat_cols


def add_engineered_features(X):
    """Create ratio and aggregate features from the original NSL-KDD columns."""
    X = X.copy()

    # Ratios/interactions. Add 1 to denominators to avoid division by zero.
    X["byte_ratio"] = X["src_bytes"] / (X["dst_bytes"] + 1)
    X["total_bytes"] = X["src_bytes"] + X["dst_bytes"]
    X["logged_in_src_bytes"] = X["logged_in"] * np.log1p(X["src_bytes"])
    X["login_failure_rate"] = X["num_failed_logins"] / (X["hot"] + 1)
    X["root_compromise_ratio"] = (X["num_root"] + X["root_shell"]) / (X["num_compromised"] + 1)

    # Aggregations over related traffic/error-rate feature groups.
    error_rate_cols = [
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]
    host_rate_cols = [
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate"
    ]
    content_cols = [
        "hot", "num_failed_logins", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "is_guest_login"
    ]

    X["mean_error_rate"] = X[error_rate_cols].mean(axis=1)
    X["max_error_rate"] = X[error_rate_cols].max(axis=1)
    X["mean_host_rate"] = X[host_rate_cols].mean(axis=1)
    X["content_activity"] = X[content_cols].sum(axis=1)

    return X


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop numeric features that are highly correlated on the training fold."""

    def __init__(self, threshold=0.98, categorical_features=None):
        self.threshold = threshold
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        X = X.copy()
        categorical = set(self.categorical_features or [])
        numeric_cols = [col for col in X.columns if col not in categorical]
        corr_matrix = X[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.features_to_drop_ = [
            col for col in upper_triangle.columns
            if any(upper_triangle[col] > self.threshold)
        ]
        return self

    def transform(self, X):
        return X.drop(columns=self.features_to_drop_, errors="ignore")


engineered_train_preview = add_engineered_features(X_train)
numeric_features = [col for col in engineered_train_preview.columns if col not in categorical_features]

print("\nFeature engineering added:")
print(sorted(set(engineered_train_preview.columns) - set(X_train.columns)))

try:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer([
    ("onehot", one_hot_encoder, make_column_selector(dtype_include=object)),
    ("scaler", StandardScaler(), make_column_selector(dtype_include=np.number))
])

mlp_pipeline = Pipeline([
    ("feature_engineering", FunctionTransformer(add_engineered_features, validate=False)),
    ("correlation_filter", CorrelationFilter(threshold=0.98, categorical_features=categorical_features)),
    ("preprocessor", preprocessor),
    ("variance_filter", VarianceThreshold()),
    ("select_k_best", SelectKBest(score_func=f_classif, k=80)),
    ("smote", SMOTE(random_state=42)),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=False,
        random_state=42
    ))
])

param_distributions = {
    "correlation_filter__threshold": [0.95, 0.98, 0.995],
    "select_k_best__k": [50, 70, 90, "all"],
    "smote__k_neighbors": [3, 5],
    "mlp__hidden_layer_sizes": [(96,), (128,), (128, 64), (160, 80)],
    "mlp__alpha": [0.0001, 0.0005, 0.001, 0.005],
    "mlp__learning_rate_init": [0.0005, 0.001, 0.002],
    "mlp__batch_size": [128, 256],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tuning_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("\nRunning RandomizedSearchCV for MLP hyperparameter tuning...")
search = RandomizedSearchCV(
    estimator=mlp_pipeline,
    param_distributions=param_distributions,
    n_iter=8,
    scoring="f1_macro",
    cv=tuning_cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
search.fit(X_train, y_train)

print("\nBest hyperparameters:")
print(search.best_params_)
print(f"Best tuning CV F1 Macro Score: {search.best_score_:.4f}")

best_mlp_pipeline = search.best_estimator_

cv_scores = cross_val_score(best_mlp_pipeline, X_train, y_train, scoring="f1_macro", cv=cv, n_jobs=-1)

print(f"MLP Cross-Validation F1 Macro Scores: {cv_scores}")
print(f"MLP Cross-Validation F1 Macro Mean Score: {cv_scores.mean():.4f}")
print(f"MLP Cross-Validation F1 Macro Std Dev: {cv_scores.std():.4f}")

best_mlp_pipeline.fit(X_train, y_train)
correlation_step = best_mlp_pipeline.named_steps["correlation_filter"]
print("\nCorrelation analysis removed redundant features:")
print(correlation_step.features_to_drop_ if correlation_step.features_to_drop_ else "None")

y_pred = best_mlp_pipeline.predict(X_test)

labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]

print("Classification Report:")
print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

print("Test Macro F1 Score:", f1_score(y_test, y_pred, average="macro"))

cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned MLP + Feature Engineering + Selection + SMOTE")
plt.tight_layout()
Path("matrixes").mkdir(exist_ok=True)
plt.savefig("matrixes/confusion_matrix_mlp_feature_tuned.png", dpi=150)
plt.show()
