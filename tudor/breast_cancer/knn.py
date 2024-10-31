import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    data = pd.read_csv('../../breast_cancer.csv')
    print(data.head()) # check everything works
    data.dropna(inplace=True)


    ### DATA PREPROCESSING ###

    # Drop id
    data.drop(['id'], axis=1, inplace=True)

    # M = 1 (Malignant), B = 0 (Benign)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Remove columns with over 50% missing values (adjust threshold if needed)
    threshold = 0.5 * data.shape[0]
    data = data.dropna(thresh=threshold, axis=1)

    # Drop remaining rows with NaNs or impute remaining values
    data.dropna(inplace=True)

    # Separate features and labels
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # Check if there’s enough data
    if X.shape[0] > 0:
        # Impute any remaining missing values just in case
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    else:
        raise ValueError("The dataset is empty after removing rows with missing values.")

    ### SPLIT DATA ###

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    ### STANDARDIZE ###

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    ### TRAIN KNN ###

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)


    ### EVALUATION ###

    # General Prediction
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Detailed classification report
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print(confusion_matrix(y_test, y_pred))
