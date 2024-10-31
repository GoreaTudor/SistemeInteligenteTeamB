import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

path = kagglehub.dataset_download("erdemtaha/cancer-data")
data_path = f"{path}/Cancer_Data.csv" 
data = pd.read_csv(data_path)

st.title("Cancer Diagnosis Prediction using SVM and Random Forest")
st.write("An interactive app to predict cancer diagnosis (Malignant or Benign) using SVM and Random Forest, with comparisons.")

data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

st.subheader("Data Visualizations")

st.write("### Pairplot of Selected Features")
sample_features = data[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']]
sns.pairplot(sample_features, hue="diagnosis", markers=["o", "s"])
st.pyplot(plt.gcf())

st.write("### Heatmap of Feature Correlations")
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap="coolwarm", annot=False)
st.pyplot(plt.gcf())

model_choice = "K-Nearest Neighbors (KNN)"

if model_choice == "K-Nearest Neighbors (KNN)":
    st.sidebar.subheader("KNN Hyperparameters")
    n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 100, value=10)


X = data.drop(columns=['diagnosis'])
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if 'trained_models' not in st.session_state:
    st.session_state['trained_models'] = []

if st.button("Train Model"):
    with st.spinner("Training the model..."):
        if model_choice == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=n_neighbors)

        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
    
    st.success("Training Complete!")
    
    st.write(f"### Model Performance for {model_choice}")
    st.write(f"**Accuracy**: {accuracy * 100:.2f}%")
    st.write("**Classification Report**:")
    st.write(pd.DataFrame(report).transpose())
    
    st.write("### Confusion Matrix")
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Benign", "Malignant"], cmap="Blues")
    st.pyplot(plt.gcf())
    
    st.session_state[f"{model_choice}_accuracy"] = accuracy
    st.session_state[f"{model_choice}_report"] = report
    st.session_state['trained_models'].append(model_choice)


if all(key in st.session_state for key in ["K-Nearest Neighbors (KNN)_accuracy"]):    
    st.write("## Model Comparison")

    comparison_data = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": []
    }

    if "K-Nearest Neighbors (KNN)_accuracy" in st.session_state:
        nb_acc = st.session_state["K-Nearest Neighbors (KNN)_accuracy"]
        nb_report = st.session_state["K-Nearest Neighbors (KNN)_report"]
        comparison_data["Model"].append("K-Nearest Neighbors (KNN)")
        comparison_data["Accuracy"].append(nb_acc)
        comparison_data["Precision"].append(nb_report["1"]["precision"]) 
        comparison_data["Recall"].append(nb_report["1"]["recall"]) 
        comparison_data["F1-Score"].append(nb_report["1"]["f1-score"])  


    comparison_df = pd.DataFrame(comparison_data)


    st.write("### Accuracy and Precision Comparison")
    fig, ax = plt.subplots(2, 1, figsize=(8, 12)) 

    comparison_df['Accuracy'] = comparison_df['Accuracy'] * 100
    comparison_df['Precision'] = comparison_df['Precision'] * 100

    sns.barplot(data=comparison_df, x='Model', y='Accuracy', ax=ax[0], palette='viridis')
    ax[0].set_title("Model Accuracy Comparison", fontsize=16)
    ax[0].set_ylabel("Accuracy (%)", fontsize=14)
    ax[0].set_ylim(0, 110)  
    ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))  

    for index, row in comparison_df.iterrows():
        ax[0].text(index, row['Accuracy'] + 5, f'{row["Accuracy"]:.1f}%', 
                color='black', ha='center', va='bottom', fontsize=12)

    sns.barplot(data=comparison_df, x='Model', y='Precision', ax=ax[1], palette='viridis')
    ax[1].set_title("Model Precision Comparison", fontsize=16)
    ax[1].set_ylabel("Precision (%)", fontsize=14)
    ax[1].set_ylim(0, 110)
    ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))  

    for index, row in comparison_df.iterrows():
        ax[1].text(index, row['Precision'] + 5, f'{row["Precision"]:.1f}%',  
                color='black', ha='center', va='bottom', fontsize=12)

    st.pyplot(fig)

    st.write("### Detailed Classification Report Comparison")
    
    for index, row in comparison_df.iterrows():
        st.write(f"**{row['Model']} Classification Report**")
        st.write(f"**Accuracy**: {row['Accuracy']:.2f}%")
        st.write(f"**Precision**: {row['Precision']:.2f}")
        st.write(f"**Recall**: {row['Recall']:.2f}")
        st.write(f"**F1-Score**: {row['F1-Score']:.2f}")
        st.write("---")  
