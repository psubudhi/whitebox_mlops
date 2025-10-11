
"""
Test Data Creation Script

Author: Prem Kumar Subudhi
Date: 11-Oct-2025
Version: 1.0

Description:
This script creates training and test datasets from a balanced dataset for machine learning projects.
It automatically detects label and text columns, splits the data into train/test sets, and saves
the results to appropriate directories.

Processing Steps:
1. DATA LOADING: Reads the balanced dataset from CSV file
2. COLUMN IDENTIFICATION: Automatically detects label and text columns using common naming conventions
3. DATA PARTITIONING: Splits dataset into training (80%) and testing (20%) subsets with stratified sampling
4. DIRECTORY MANAGEMENT: Creates required output directories if they don't exist
5. DATA PERSISTENCE: Saves processed datasets to CSV files in organized directory structure
6. REPORTING: Generates comprehensive summary statistics and distribution analysis

Dependencies:
- pandas
- scikit-learn
"""

# create_test_data.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def create_test_data():
    """
    Create training and test datasets from balanced dataset.

    This function:
    1. Loads the balanced dataset from CSV
    2. Auto-detects label and text columns using common naming conventions
    3. Splits the data into training (80%) and test (20%) sets with stratification
    4. Saves the resulting datasets to appropriate directories

    Returns:
        bool: True if successful, False if column detection fails

    Raises:
        FileNotFoundError: If balanced_dataset.csv doesn't exist
        Exception: For other unexpected errors during processing
    """

    try:
        # Load balanced dataset
        print("Loading balanced dataset...")
        balanced_df = pd.read_csv("balanced_dataset/balanced_dataset.csv")

        # Auto-detect columns using common naming conventions
        label_candidates = ['label', 'category', 'class', 'target', 'industry', 'industry_name']
        text_candidates = ['text', 'content', 'summary', 'description', 'full_summary']

        label_col = None
        text_col = None

        # Detect label column
        for candidate in label_candidates:
            if candidate in balanced_df.columns:
                label_col = candidate
                break

        # Detect text column
        for candidate in text_candidates:
            if candidate in balanced_df.columns:
                text_col = candidate
                break

        # Validate column detection
        if not label_col or not text_col:
            print("Error: Could not auto-detect required columns")
            print(f"Available columns: {balanced_df.columns.tolist()}")
            print(f"Label candidates tried: {label_candidates}")
            print(f"Text candidates tried: {text_candidates}")
            return False

        print(f"Using label column: {label_col}")
        print(f"Using text column: {text_col}")

        # Split into train and test sets with stratification
        print("Splitting data into train and test sets...")
        train_df, test_df = train_test_split(
            balanced_df,
            test_size=0.2,
            random_state=42,
            stratify=balanced_df[label_col]
        )

        # Create directories if they don't exist
        os.makedirs("train_test_data", exist_ok=True)
        os.makedirs("balanced_dataset", exist_ok=True)

        # Save train and test data
        train_df.to_csv("train_test_data/train_data.csv", index=False)
        test_df.to_csv("train_test_data/test_data.csv", index=False)
        balanced_df.to_csv("balanced_dataset/balanced_dataset.csv", index=False)

        # Print summary statistics
        print("\n=== Data Split Summary ===")
        print(f"Original data shape: {balanced_df.shape}")
        print(f"Train data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        print(f"Train data saved to: train_test_data/train_data.csv")
        print(f"Test data saved to: train_test_data/test_data.csv")

        # Show class distribution
        if label_col:
            print(f"\nClass distribution in training set:")
            print(train_df[label_col].value_counts())
            print(f"\nClass distribution in test set:")
            print(test_df[label_col].value_counts())

        return True

    except FileNotFoundError:
        print("Error: balanced_dataset/balanced_dataset.csv not found")
        print("Please ensure the balanced dataset exists in the correct location")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    """
    Main execution block when script is run directly
    """
    print("Starting test data creation...")
    success = create_test_data()

    if success:
        print("\nTest data creation completed successfully!")
    else:
        print("\nTest data creation failed!")
        exit(1)

from google.colab import drive
drive.mount('/content/drive')

