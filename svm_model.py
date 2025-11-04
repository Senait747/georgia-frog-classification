import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import wandb

# Initialize W&B
wandb.init(project="Georgia-Frog-ID-SVM", config={
    "model_type": "SVM",
    "kernel": "rbf",
    "feature_count": 38,
    "species_classes": 15,
    "train_samples": 979,
    "val_samples": 210
})

print("=== Loading Data for SVM ===")
# Load features
features = pd.read_csv('processed/spectrograms/features_aggregated.csv')
train_meta = pd.read_csv('processed/spectrograms/metadata_train.csv')
val_meta = pd.read_csv('processed/spectrograms/metadata_val.csv')

# Prepare features (exclude non-numerical columns)
feature_columns = [col for col in features.columns if col not in ['file_id', 'species', 'processed_path', 'split', 'provenance']]
print(f"Using {len(feature_columns)} numerical features")

# Split data
X_train = features[features['file_id'].isin(train_meta['file_id'])][feature_columns]
y_train = features[features['file_id'].isin(train_meta['file_id'])]['species']

X_val = features[features['file_id'].isin(val_meta['file_id'])][feature_columns]
y_val = features[features['file_id'].isin(val_meta['file_id'])]['species']

print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

# Scale features (important for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("=== Training SVM ===")
# Train SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)

print("=== Evaluating SVM ===")
# Predictions
y_pred = svm.predict(X_val_scaled)

# Calculate metrics
accuracy = accuracy_score(y_val, y_pred)
f1_macro = f1_score(y_val, y_pred, average='macro')
f1_weighted = f1_score(y_val, y_pred, average='weighted')

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Macro F1: {f1_macro:.4f}")
print(f"Weighted F1: {f1_weighted:.4f}")

# Log to W&B
wandb.log({
    "val_accuracy": accuracy,
    "val_f1_macro": f1_macro,
    "val_f1_weighted": f1_weighted
})

# Log per-class metrics
class_report = classification_report(y_val, y_pred, output_dict=True)
for species, metrics in class_report.items():
    if species not in ['accuracy', 'macro avg', 'weighted avg']:
        wandb.log({
            f"precision_{species}": metrics['precision'],
            f"recall_{species}": metrics['recall'],
            f"f1_{species}": metrics['f1-score']
        })

print("\n=== Classification Report ===")
print(classification_report(y_val, y_pred))

# Log confusion matrix
wandb.sklearn.plot_confusion_matrix(y_val, y_pred, labels=sorted(y_val.unique()))

wandb.finish()
print("=== SVM Training Complete! ===")