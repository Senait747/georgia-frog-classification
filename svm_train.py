import pandas as pd
import os

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
    print("\nFirst 5 rows:")
    print(features.head())
    
    print(f"\nData types:")
    print(features.dtypes)
    
    # Check if there are numerical features
    numerical_cols = features.select_dtypes(include=['number']).columns
    print(f"\nNumerical columns: {list(numerical_cols)}")
    
    # Check for species column
    if 'species' in features.columns:
        print(f"\nSpecies distribution:")
        print(features['species'].value_counts())
        
except Exception as e:
    print(f"Error reading features file: {e}")

print("\n=== Checking metadata files ===")
try:
    train_meta = pd.read_csv('processed/spectrograms/metadata_train.csv')
    val_meta = pd.read_csv('processed/spectrograms/metadata_val.csv')
    
    print(f"Train metadata shape: {train_meta.shape}")
    print(f"Val metadata shape: {val_meta.shape}")
    print(f"Train columns: {list(train_meta.columns)}")
    
except Exception as e:
    print(f"Error reading metadata: {e}")
