# Step 1: Loading Libraries and Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load the data (replace 'breast_cancer_data.csv' with the correct file path)
    df = pd.read_csv('../../breast_cancer.csv')

    # Display the first few rows
    df.head()

    # Step 2: Basic Information
    print("Shape of dataset:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nChecking for missing values:\n", df.isnull().sum())

    # Step 3: Summary Statistics
    print("\nSummary Statistics:\n", df.describe())

    # Step 4: Distribution of Target Variable (diagnosis)
    plt.figure(figsize=(6, 4))
    df['diagnosis'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black')
    plt.title("Diagnosis Distribution (Benign vs Malignant)")
    plt.xlabel("Diagnosis (M: Malignant, B: Benign)")
    plt.ylabel("Count")
    plt.show()

    # Step 5: Feature Distribution Analysis
    # Dropping 'id' as it’s not useful for analysis
    df.drop(columns='id', inplace=True)

    # Encoding diagnosis as numeric: M = 1, B = 0 for correlation purposes
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

    # Plotting histograms for numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    num_features = len(numerical_features)

    # Calculate appropriate layout
    rows = (num_features // 4) + 1 if num_features % 4 != 0 else num_features // 4
    fig, axes = plt.subplots(rows, 4, figsize=(20, 15))
    axes = axes.flatten()

    for i, feature in enumerate(numerical_features):
        df[feature].hist(bins=15, color='teal', edgecolor='black', ax=axes[i])
        axes[i].set_title(feature)

    # Hide any empty subplots
    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()

    # Step 6: Correlation Matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
    plt.title("Feature Correlation Matrix")
    plt.show()

    # Step 7: Top Correlations with Diagnosis
    # Find features most correlated with diagnosis
    correlation_with_diagnosis = correlation_matrix['diagnosis'].sort_values(ascending=False)
    print("\nTop features correlated with diagnosis:\n", correlation_with_diagnosis[1:6])

    # Step 8: Checking Class Imbalance
    # Ratio of malignant to benign cases
    malignant_ratio = df['diagnosis'].value_counts(normalize=True)[1] * 100  # '1' represents Malignant after encoding
    print(f"\nMalignant cases ratio: {malignant_ratio:.2f}%")
