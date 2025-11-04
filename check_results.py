import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("=== Loading Data ===")
features = pd.read_csv('processed/spectrograms/features_aggregated.csv')
train_meta = pd.read_csv('processed/spectrograms/metadata_train.csv')
val_meta = pd.read_csv('processed/spectrograms/metadata_val.csv')

feature_columns = [col for col in features.columns if col not in ['file_id', 'species', 'processed_path', 'split', 'provenance']]

print("=== Preparing Data ===")
X_train = features[features['file_id'].isin(train_meta['file_id'])][feature_columns]
y_train = features[features['file_id'].isin(train_meta['file_id'])]['species']
X_val = features[features['file_id'].isin(val_meta['file_id'])][feature_columns]
y_val = features[features['file_id'].isin(val_meta['file_id'])]['species']

print("=== Training SVM ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_val_scaled)

print("=== RESULTS ===")
print(f"Validation Accuracy: {(y_pred == y_val).mean():.4f}")
print(f"Validation samples: {len(y_val)}")
print(f"Species in validation: {y_val.nunique()}")
print("\n=== Classification Report ===")
print(classification_report(y_val, y_pred))

print("\n=== Species Performance ===")
report = classification_report(y_val, y_pred, output_dict=True)
for species in sorted(y_val.unique()):
    if species in report:
        print(f"{species:25} F1: {report[species]['f1-score']:.3f}")

# Create confusion matrix visualization
print("\n=== Creating Confusion Matrix ===")
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y_val.unique()), 
            yticklabels=sorted(y_val.unique()))
plt.title('SVM Confusion Matrix - Frog Species Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")