# svm_train_improved.py - COMPLETE IMPROVED VERSION

import pandas as pd
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=== COMPLETE FROG SPECIES CLASSIFICATION ===")

# =============================================================================
# STEP 1: DATA VALIDATION
# =============================================================================

print("=== Checking if files exist ===")
files_to_check = [
    'processed/spectrograms/features_aggregated.csv',
    'processed/spectrograms/metadata_train.csv', 
    'processed/spectrograms/metadata_val.csv',
    'processed/spectrograms/metadata_test.csv'
]

for file in files_to_check:
    if os.path.exists(file):
        print(f"✅ {file} - EXISTS")
    else:
        print(f"❌ {file} - MISSING")

print("\n=== Checking features_aggregated.csv ===")
try:
    # Load the features file
    features = pd.read_csv('processed/spectrograms/features_aggregated.csv')
    
    print(f"Shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    
    # Check for species column
    if 'species' in features.columns:
        print(f"\nSpecies distribution:")
        species_counts = features['species'].value_counts()
        print(species_counts)
        
except Exception as e:
    print(f"Error reading features file: {e}")

print("\n=== Checking metadata files ===")
try:
    train_meta = pd.read_csv('processed/spectrograms/metadata_train.csv')
    val_meta = pd.read_csv('processed/spectrograms/metadata_val.csv')
    test_meta = pd.read_csv('processed/spectrograms/metadata_test.csv')
    
    print(f"Train metadata shape: {train_meta.shape}")
    print(f"Val metadata shape: {val_meta.shape}")
    print(f"Test metadata shape: {test_meta.shape}")
    
except Exception as e:
    print(f"Error reading metadata: {e}")

# =============================================================================
# STEP 2: IMPROVED SVM TRAINING WITH CLASS WEIGHTS
# =============================================================================

def train_improved_svm():
    """Train SVM with fixes for poorly identified species"""
    
    print("\n" + "="*60)
    print("TRAINING IMPROVED SVM MODEL")
    print("="*60)
    
    try:
        # Load and merge data
        features = pd.read_csv('processed/spectrograms/features_aggregated.csv')
        train_meta = pd.read_csv('processed/spectrograms/metadata_train.csv')
        val_meta = pd.read_csv('processed/spectrograms/metadata_val.csv')
        test_meta = pd.read_csv('processed/spectrograms/metadata_test.csv')
        
        # Merge datasets
        train_data = train_meta.merge(features, left_on='filename', right_on='file_id', how='inner')
        val_data = val_meta.merge(features, on='filename', how='inner')
        test_data = test_meta.merge(features, on='filename', how='inner')
        
        # Prepare features and labels
        feature_cols = [col for col in features.columns if col not in ['file_id', 'species', 'processed_path', 'split', 'provenance']]
        
        X_train = train_data[feature_cols]
        y_train = train_data['species']
        X_val = val_data[feature_cols]
        y_val = val_data['species']
        X_test = test_data[feature_cols]
        y_test = test_data['species']
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # =====================================================================
        # KEY IMPROVEMENT: Handle class imbalance
        # =====================================================================
        
        # Analyze species distribution
        print("\n=== SPECIES ANALYSIS ===")
        species_counts = y_train.value_counts()
        for species, count in species_counts.items():
            status = "⚠️ LOW SAMPLES" if count < 10 else "✅ OK"
            print(f"  {species}: {count} samples - {status}")
        
        # Compute class weights to handle imbalance
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        print(f"\nClass weights for imbalance: {class_weights}")
        
        # =====================================================================
        # Preprocessing
        # =====================================================================
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)
        
        # =====================================================================
        # Train SVM with class weights
        # =====================================================================
        print("\n=== TRAINING SVM WITH CLASS WEIGHTS ===")
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale', 
            class_weight=class_weights,  # KEY FIX for poor species identification
            random_state=42,
            probability=True
        )
        
        svm_model.fit(X_train_scaled, y_train_encoded)
        
        # =====================================================================
        # Evaluate model
        # =====================================================================
        print("\n=== EVALUATION ===")
        
        # Training accuracy
        train_pred = svm_model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train_encoded, train_pred)
        print(f"Training Accuracy: {train_acc:.4f}")
        
        # Validation accuracy
        val_pred = svm_model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val_encoded, val_pred)
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        # Test accuracy
        test_pred = svm_model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test_encoded, test_pred)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Per-species performance
        print("\n=== PER-SPECIES PERFORMANCE ===")
        unique_species = label_encoder.classes_
        
        for i, species in enumerate(unique_species):
            species_mask = y_test_encoded == i
            if np.sum(species_mask) > 0:
                species_accuracy = accuracy_score(
                    y_test_encoded[species_mask], 
                    test_pred[species_mask]
                )
                samples = np.sum(species_mask)
                status = "✅ GOOD" if species_accuracy > 0.7 else "⚠️ NEEDS IMPROVEMENT" if species_accuracy > 0.4 else "❌ POOR"
                print(f"  {species}: {species_accuracy:.3f} ({samples} samples) - {status}")
        
        # =====================================================================
        # Save model
        # =====================================================================
        os.makedirs('saved_models', exist_ok=True)
        model_data = {
            'model': svm_model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_columns': feature_cols
        }
        joblib.dump(model_data, 'saved_models/improved_svm_model.pkl')
        print("\n✅ Model saved to 'saved_models/improved_svm_model.pkl'")
        
        # =====================================================================
        # Plot results
        # =====================================================================
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test_encoded, test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix (Test Accuracy: {test_acc:.4f})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n🎯 IMPROVEMENTS MADE:")
        print("1. Added class_weight='balanced' to handle species imbalance")
        print("2. Used StandardScaler for feature normalization") 
        print("3. Added per-species performance analysis")
        print("4. Saved model with all preprocessing components")
        
        return svm_model, test_acc
        
    except Exception as e:
        print(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

# Run the improved training
if __name__ == "__main__":
    model, accuracy = train_improved_svm()
    if model is not None:
        print(f"\n🎉 Training completed! Final Test Accuracy: {accuracy:.4f}")
        print("\n📁 Files created:")
        print("   - saved_models/improved_svm_model.pkl")
        print("   - improved_confusion_matrix.png")
    else:
        print("\n❌ Training failed!")
