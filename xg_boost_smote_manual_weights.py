from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTENC
import numpy as np

def add_nsl_kdd_features(X):
    X = X.copy()

    # Byte behavior
    X["total_bytes"] = X["src_bytes"] + X["dst_bytes"]
    X["byte_diff"] = X["src_bytes"] - X["dst_bytes"]
    X["byte_ratio"] = (X["src_bytes"] + 1) / (X["dst_bytes"] + 1)

    X["byte_asymmetry"] = (
        (X["src_bytes"] - X["dst_bytes"]) /
        (X["src_bytes"] + X["dst_bytes"] + 1)
    )

    X["log_src_bytes"] = np.log1p(X["src_bytes"])
    X["log_dst_bytes"] = np.log1p(X["dst_bytes"])
    X["log_total_bytes"] = np.log1p(X["total_bytes"])

    X["zero_src_bytes"] = (X["src_bytes"] == 0).astype(int)
    X["zero_dst_bytes"] = (X["dst_bytes"] == 0).astype(int)

    # R2L / U2R access-risk features
    X["login_risk"] = (
        X["num_failed_logins"]
        + X["is_guest_login"]
        + X["num_access_files"]
    )

    X["privilege_risk"] = (
        X["root_shell"]
        + X["su_attempted"]
        + X["num_shells"]
        + X["num_root"]
    )

    X["compromise_activity"] = (
        X["hot"]
        + X["num_compromised"]
        + X["num_file_creations"]
    )

    X["access_content_risk"] = (
        X["login_risk"]
        + X["privilege_risk"]
        + X["compromise_activity"]
    )

    # DoS / Probe traffic-pattern features
    X["serror_mean"] = (
        X["serror_rate"]
        + X["srv_serror_rate"]
        + X["dst_host_serror_rate"]
        + X["dst_host_srv_serror_rate"]
    ) / 4

    X["rerror_mean"] = (
        X["rerror_rate"]
        + X["srv_rerror_rate"]
        + X["dst_host_rerror_rate"]
        + X["dst_host_srv_rerror_rate"]
    ) / 4

    X["same_service_pressure"] = (
        X["same_srv_rate"]
        + X["dst_host_same_srv_rate"]
    ) / 2

    X["different_service_pressure"] = (
        X["diff_srv_rate"]
        + X["dst_host_diff_srv_rate"]
    ) / 2

    X["srv_count_ratio"] = X["srv_count"] / (X["count"] + 1)
    X["host_service_ratio"] = X["dst_host_srv_count"] / (X["dst_host_count"] + 1)

    return X

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
# 2. ENCODE CATEGORICAL FEATURES
# ==============================================================

# Merge temporarily to ensure consistent encoding across train and test
df_full = pd.concat([df_train, df_test])

# Encode categorical columns as integers
cat_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])
    label_encoders[col] = le

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

X_train = add_nsl_kdd_features(X_train)
X_test = add_nsl_kdd_features(X_test)

print(f"\nFeatures: {X_train.shape[1]}")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())

# ==============================================================
# YOUR WORK STARTS HERE
# ==============================================================
# 
# You now have:
#   X_train, y_train  — training features and labels (5 categories)
#   X_test, y_test    — test features and labels (5 categories)
#
# Your task:
#   1. Train one or more models on X_train / y_train
#   2. Predict on X_test
#   3. Evaluate using macro F1-score
#
# #   f1_score(y_test, y_pred, average='macro')Useful imports for evaluation:
#   from sklearn.metrics import classification_report, confusion_matrix, f1_score
#
# To compute macro F1:
#   f1_score(y_test, y_pred, average='macro')

# ==============================================================
# 5. ENCODE TARGET LABELS
# ==============================================================

target_encoder = LabelEncoder()
y_train_enc = target_encoder.fit_transform(y_train)
y_test_enc = target_encoder.transform(y_test)

print("Target classes:", target_encoder.classes_)

# ==============================================================
# 6. DEFINE CATEGORICAL / NUMERICAL COLUMNS
# ==============================================================

cat_cols = ["protocol_type", "service", "flag"]
num_cols = [col for col in X_train.columns if col not in cat_cols]

cat_indices = [X_train.columns.get_loc(col) for col in cat_cols]

# ==============================================================
# 7. APPLY SMOTENC
# ==============================================================

smote = SMOTENC(
    categorical_features=cat_indices,
    sampling_strategy="not majority",
    k_neighbors=3,
    random_state=42
)

X_res, y_res = smote.fit_resample(X_train, y_train_enc)

# SMOTENC may return a NumPy array, so convert it back to DataFrame
X_res = pd.DataFrame(X_res, columns=X_train.columns)

# Make sure categorical columns are treated as categorical IDs
for col in cat_cols:
    X_res[col] = X_res[col].astype(int)
    X_test[col] = X_test[col].astype(int)

# ==============================================================
# 8. ONE-HOT ENCODE CATEGORICAL FEATURES
# ==============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

X_res_encoded = preprocessor.fit_transform(X_res)
X_test_encoded = preprocessor.transform(X_test)

print("Shape after one-hot encoding:")
print("Train:", X_res_encoded.shape)
print("Test:", X_test_encoded.shape)

# ==============================================================
# 9. CREATE SAMPLE WEIGHTS AFTER SMOTE
# ==============================================================

class_weight_map = {
    target_encoder.transform(["Normal"])[0]: 1.0,
    target_encoder.transform(["DoS"])[0]: 1.3,
    target_encoder.transform(["Probe"])[0]: 1.5,
    target_encoder.transform(["R2L"])[0]: 8.0,
    target_encoder.transform(["U2R"])[0]: 20.0,
}

sample_weights = np.ones(len(y_res), dtype=float)

for class_id, weight in class_weight_map.items():
    sample_weights[y_res == class_id] = weight

# ==============================================================
# 10. TRAIN XGBOOST
# ==============================================================

params = {
    "subsample": 1.0,
    "reg_lambda": 10,
    "n_estimators": 500,
    "min_child_weight": 3,
    "max_depth": 6,
    "learning_rate": 0.05,
    "gamma": 0.1,
    "colsample_bytree": 0.7,
}

model = XGBClassifier(
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    **params
)

model.fit(
    X_res_encoded,
    y_res,
    sample_weight=sample_weights
)

y_pred_enc = model.predict(X_test_encoded)
y_pred = target_encoder.inverse_transform(y_pred_enc)

# ==============================================================
# 11. R2L SPECIALIST MODEL: NORMAL VS R2L
# ==============================================================

binary_mask = y_train.isin(["Normal", "R2L"])

X_train_binary = X_train.copy()

y_train_binary = (y_train == "R2L").astype(int)

# Same categorical / numerical split
cat_cols = ["protocol_type", "service", "flag"]
num_cols = [col for col in X_train_binary.columns if col not in cat_cols]

# One-hot preprocessing for the binary specialist
r2l_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

X_train_binary_encoded = r2l_preprocessor.fit_transform(X_train_binary)

normal_count = (y_train_binary == 0).sum()
r2l_count = (y_train_binary == 1).sum()

neg_count = (y_train_binary == 0).sum()
r2l_count = (y_train_binary == 1).sum()

scale_pos_weight = neg_count / r2l_count

r2l_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

r2l_model.fit(X_train_binary_encoded, y_train_binary)

# ==============================================================
# 12. CHAINED PREDICTION
# ==============================================================

final_pred = y_pred.copy()

normal_indices = np.where(y_pred == "Normal")[0]

X_test_normal_candidates = X_test.iloc[normal_indices].copy()
X_test_normal_candidates_encoded = r2l_preprocessor.transform(X_test_normal_candidates)

r2l_proba = r2l_model.predict_proba(X_test_normal_candidates_encoded)[:, 1]

threshold = 0.2

override_indices = normal_indices[r2l_proba >= threshold]

final_pred[override_indices] = "R2L"

print("Number of Normal -> R2L overrides:", len(override_indices))

print("Macro F1 with chained R2L specialist:", f1_score(y_test, final_pred, average="macro"))
print(classification_report(y_test, final_pred))

# ==============================================================
# 13. SAVE CONFUSION MATRIX FOR CHAINED MODEL
# ==============================================================

labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]

cm = confusion_matrix(y_test, final_pred, labels=labels)

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
plt.title("Confusion Matrix - Chained XGBoost + R2L Specialist")
plt.tight_layout()

plt.savefig("confusion_matrix_chained_r2l.png", dpi=150)
plt.close()

# ==============================================================
# 14. CROSS-VALIDATION (MACRO F1)
# ==============================================================

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

print("\nRunning 5-fold cross-validation on training data...")

cv_pipeline = Pipeline([
    ("preprocessor", ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", [col for col in X_train.columns if col not in cat_cols])
        ]
    )),
    ("classifier", XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        **params
    ))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train_enc_full = target_encoder.transform(y_train)

cv_f1_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_enc_full), 1):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train_enc_full[train_idx], y_train_enc_full[val_idx]

    cv_pipeline.fit(X_fold_train, y_fold_train)
    y_fold_pred = cv_pipeline.predict(X_fold_val)
    fold_f1 = f1_score(y_fold_val, y_fold_pred, average="macro")
    cv_f1_scores.append(fold_f1)
    print(f"  Fold {fold}: Macro F1 = {fold_f1:.4f}")

print(f"\nCross-Validation Macro F1: {np.mean(cv_f1_scores):.4f} +/- {np.std(cv_f1_scores):.4f}")





