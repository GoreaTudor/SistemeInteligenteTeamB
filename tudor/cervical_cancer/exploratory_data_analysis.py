# Step 1: Loading Libraries and Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load the data, treating '?' as NaN
    df = pd.read_csv("..\\..\\cervical_cancer.csv", na_values="?")

    # Display the first few rows
    df.head()

    # Step 2: Basic Information
    print("Shape of dataset:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nChecking for missing values:\n", df.isnull().sum())

    # Step 3: Summary Statistics
    print("\nSummary Statistics:\n", df.describe())

    # Step 4: Distribution of Target Variable (assuming 'Biopsy' is the target)
    plt.figure(figsize=(6, 4))
    df['Biopsy'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black')
    plt.title("Biopsy Results Distribution")
    plt.xlabel("Biopsy Result (1: Positive, 0: Negative)")
    plt.ylabel("Count")
    plt.show()

    # Step 5: Handling Missing Values
    # Check for columns with missing values
    missing_values = df.isnull().sum()
    print("\nColumns with missing values:\n", missing_values[missing_values > 0])

    # Fill numerical missing values with median (for simplicity)
    numerical_features = df.select_dtypes(include=[np.number]).columns
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())

    # For categorical data, we can fill NaNs with the mode or drop rows if appropriate
    # Uncomment the following line to drop rows with missing target values, if any.
    # df = df.dropna(subset=['Biopsy'])

    # Step 6: Feature Distribution Analysis
    num_features = len(numerical_features)

    # Calculate appropriate layout for subplots
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

    # Step 7: Correlation Matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
    plt.title("Feature Correlation Matrix")
    plt.show()

    # Step 8: Top Correlations with Biopsy Result
    correlation_with_biopsy = correlation_matrix['Biopsy'].sort_values(ascending=False)
    print("\nTop features correlated with Biopsy result:\n", correlation_with_biopsy[1:6])

    # Step 9: Checking Class Imbalance
    positive_ratio = df['Biopsy'].value_counts(normalize=True)[1] * 100
    print(f"\nPositive biopsy cases ratio: {positive_ratio:.2f}%")
