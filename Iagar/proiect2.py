# Import necessary libraries
import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import time

# Download the latest version of the dataset
path = kagglehub.dataset_download("erdemtaha/cancer-data")
data_path = f"{path}/Cancer_Data.csv"  # Adjust path if necessary
data = pd.read_csv(data_path)

# Streamlit title and description
st.title("Cancer Diagnosis Prediction using SVM and Random Forest")
st.write("An interactive app to predict cancer diagnosis (Malignant or Benign) using SVM and Random Forest, with comparisons.")

# Data Preprocessing
data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Sidebar for dataset exploration
st.sidebar.header("Dataset Exploration")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(data)

# Data Visualization
st.subheader("Data Visualizations")

# Pairplot for selected features
st.write("### Pairplot of Selected Features")
sample_features = data[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']]
sns.pairplot(sample_features, hue="diagnosis", markers=["o", "s"])
st.pyplot(plt.gcf())

# Heatmap of feature correlations
st.write("### Heatmap of Feature Correlations")
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap="coolwarm", annot=False)
st.pyplot(plt.gcf())

# Model Selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose Model", ["Support Vector Machine (SVM)", "Random Forest", "Naive Bayes"])

# Algorithm information popups
if model_choice == "Support Vector Machine (SVM)":
    if st.sidebar.button("Learn More about SVM"):
        st.sidebar.info("""
        **Support Vector Machine (SVM)**  
        SVM is a supervised machine learning algorithm that aims to find a hyperplane that best separates classes in a dataset.  
        It works by maximizing the margin between data points of different classes, making it well-suited for binary classification tasks.  
        Common hyperparameters include:  
        - **C**: Regularization parameter that controls the trade-off between maximizing the margin and minimizing classification error.  
        - **Kernel**: Defines the function used to transform data before separating it; common options include linear, polynomial, and radial basis function (RBF) kernels.
        """)

    # SVM Hyperparameters
    st.sidebar.subheader("SVM Hyperparameters")
    C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, value=1.0, step=0.01, help="Controls trade-off between smooth decision boundary and correct classification.")
    kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], help="Specifies the kernel type for SVM.")
    degree = st.sidebar.slider("Degree", 1, 5, value=3, help="Degree of the polynomial kernel (used only with 'poly' kernel).") if kernel == "poly" else 3
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"], help="Kernel coefficient. 'Scale' uses (1 / n_features); 'auto' uses 1 / n_samples.")

elif model_choice == "Random Forest":
    if st.sidebar.button("Learn More about Random Forest"):
        st.sidebar.info("""
        **Random Forest**  
        Random Forest is an ensemble learning algorithm that builds multiple decision trees on different samples of the data.  
        It combines the predictions of each tree to produce a final classification result, improving accuracy and reducing overfitting.  
        Key hyperparameters include:  
        - **n_estimators**: The number of trees in the forest.  
        - **max_depth**: The maximum depth of each tree, limiting its growth and complexity.  
        - **min_samples_split** and **min_samples_leaf**: Control when nodes should split, helping to prevent overfitting.
        """)

    # Random Forest Hyperparameters
    st.sidebar.subheader("Random Forest Hyperparameters")
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, value=100, step=10, help="Number of trees in the forest.")
    max_depth = st.sidebar.slider("Max Depth", 1, 20, value=10, help="Maximum depth of each tree.")
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, value=2, help="Minimum samples required to split an internal node.")
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, value=1, help="Minimum samples required to be at a leaf node.")

elif model_choice == "Naive Bayes":
    if st.sidebar.button("Learn More about Naive Bayes"):
        st.sidebar.info("""
        **Naive Bayes**  
        Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions.  
        It is particularly effective for high-dimensional datasets and is widely used for classification tasks.  
        Key aspects include:  
        - **GaussianNB**: Assumes that the features follow a normal distribution.
        """)

# Splitting and Standardizing Data
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize session state for models
if 'trained_models' not in st.session_state:
    st.session_state['trained_models'] = []

# Training Button with Progress Bar
if st.button("Train Model"):
    with st.spinner("Training the model..."):
        progress_bar = st.progress(0)
        for pct in range(1, 101):
            time.sleep(0.01)
            progress_bar.progress(pct)

        # Initialize and train the selected model
        if model_choice == "Support Vector Machine (SVM)":
            model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf, random_state=42)
        elif model_choice == "Naive Bayes":
            model = GaussianNB()  # Initialize Gaussian Naive Bayes model
        
        model.fit(X_train, y_train)
        
        # Make predictions and evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
    
    st.success("Training Complete!")
    
    # Display accuracy and classification report
    st.write(f"### Model Performance for {model_choice}")
    st.write(f"**Accuracy**: {accuracy * 100:.2f}%")
    st.write("**Classification Report**:")
    st.write(pd.DataFrame(report).transpose())
    
    # Confusion Matrix
    st.write("### Confusion Matrix")
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Benign", "Malignant"], cmap="Blues")
    st.pyplot(plt.gcf())
    
    # Save comparison details for later analysis
    st.session_state[f"{model_choice}_accuracy"] = accuracy
    st.session_state[f"{model_choice}_report"] = report
    st.session_state['trained_models'].append(model_choice)


# Always display trained models section with empty checkboxes
st.sidebar.header("Trained Models")
st.sidebar.write("Models trained so far:")
for model in ["Support Vector Machine (SVM)", "Random Forest", "Naive Bayes"]:
    # Set checkbox based on whether the model has been trained
    is_trained = model in st.session_state['trained_models']
    st.sidebar.checkbox(model, value=is_trained, disabled=True)

# Model Comparison with Visualization
if "Support Vector Machine (SVM)_accuracy" in st.session_state and "Random Forest_accuracy" in st.session_state  and "Naive Bayes_accuracy" in st.session_state:
    st.write("## Model Comparison")

    # Initialize the comparison DataFrame
    comparison_data = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": []
    }

    # Populate the comparison data for SVM
    if "Support Vector Machine (SVM)_accuracy" in st.session_state:
        svm_acc = st.session_state["Support Vector Machine (SVM)_accuracy"]
        svm_report = st.session_state["Support Vector Machine (SVM)_report"]
        comparison_data["Model"].append("SVM")
        comparison_data["Accuracy"].append(svm_acc)
        comparison_data["Precision"].append(svm_report['1']['precision'])  # Assuming '1' is malignant
        comparison_data["Recall"].append(svm_report['1']['recall'])
        comparison_data["F1-Score"].append(svm_report['1']['f1-score'])

    # Populate the comparison data for Random Forest
    if "Random Forest_accuracy" in st.session_state:
        rf_acc = st.session_state["Random Forest_accuracy"]
        rf_report = st.session_state["Random Forest_report"]
        comparison_data["Model"].append("Random Forest")
        comparison_data["Accuracy"].append(rf_acc)
        comparison_data["Precision"].append(rf_report['1']['precision'])
        comparison_data["Recall"].append(rf_report['1']['recall'])
        comparison_data["F1-Score"].append(rf_report['1']['f1-score'])

     # Populate the comparison data for Naive Bayes
    if "Naive Bayes_accuracy" in st.session_state:
        nb_acc = st.session_state["Naive Bayes_accuracy"]
        nb_report = st.session_state["Naive Bayes_report"]
        comparison_data["Model"].append("Naive Bayes")
        comparison_data["Accuracy"].append(nb_acc)
        comparison_data["Precision"].append(nb_report["1"]["precision"])  # Malignant precision
        comparison_data["Recall"].append(nb_report["1"]["recall"])  # Malignant recall
        comparison_data["F1-Score"].append(nb_report["1"]["f1-score"])  # Malignant F1


    # Create a DataFrame for plotting
    comparison_df = pd.DataFrame(comparison_data)


    # Plotting the accuracies
    st.write("### Accuracy and Precision Comparison")
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))  # Create subplots for accuracy and precision

    # Multiply accuracy and precision values by 100 for percentage display
    comparison_df['Accuracy'] = comparison_df['Accuracy'] * 100
    comparison_df['Precision'] = comparison_df['Precision'] * 100

    # Create the bar plot for accuracy
    sns.barplot(data=comparison_df, x='Model', y='Accuracy', ax=ax[0], palette='viridis')
    ax[0].set_title("Model Accuracy Comparison", fontsize=16)
    ax[0].set_ylabel("Accuracy (%)", fontsize=14)
    ax[0].set_ylim(0, 110)  # Set y-axis limit to show accuracy in percentage
    ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))  # Format y-axis as percentage

    # Annotate bars with accuracy values
    for index, row in comparison_df.iterrows():
        ax[0].text(index, row['Accuracy'] + 5, f'{row["Accuracy"]:.1f}%',  # Adjusted from +2 to +5
                color='black', ha='center', va='bottom', fontsize=12)

    # Create the bar plot for precision
    sns.barplot(data=comparison_df, x='Model', y='Precision', ax=ax[1], palette='viridis')
    ax[1].set_title("Model Precision Comparison", fontsize=16)
    ax[1].set_ylabel("Precision (%)", fontsize=14)
    ax[1].set_ylim(0, 110)  # Set y-axis limit to show precision in percentage
    ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))  # Format y-axis as percentage

    # Annotate bars with precision values
    for index, row in comparison_df.iterrows():
        ax[1].text(index, row['Precision'] + 5, f'{row["Precision"]:.1f}%',  # Adjusted from +2 to +5
                color='black', ha='center', va='bottom', fontsize=12)

    st.pyplot(fig)

    # Detailed Classification Report Comparison
    st.write("### Detailed Classification Report Comparison")
    
    for index, row in comparison_df.iterrows():
        st.write(f"**{row['Model']} Classification Report**")
        st.write(f"**Accuracy**: {row['Accuracy']:.2f}%")
        st.write(f"**Precision**: {row['Precision']:.2f}")
        st.write(f"**Recall**: {row['Recall']:.2f}")
        st.write(f"**F1-Score**: {row['F1-Score']:.2f}")
        st.write("---")  # Divider line for clarity
