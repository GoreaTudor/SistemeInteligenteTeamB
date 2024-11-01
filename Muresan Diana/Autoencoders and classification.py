import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam

# -extract data-
data = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Cancer_Data.csv')
data = data.dropna(axis=1, how='any')

# -separate target-
target = data['diagnosis'].map({'M': 1, 'B': 0})  #'M' for malignant, 'B' for benign
data = data.drop(['diagnosis', 'id'], axis=1)

# -normalize-
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# -training and testing sets-
X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.2, random_state=42)

# -autoencoder architecture-
input_dim = X_train.shape[1]

# -encoder-
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(64, activation='relu')(input_layer)
encoded = layers.Dense(32, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)
latent_space = layers.Dense(8, activation='relu')(encoded)

# -decoder-
decoded = layers.Dense(32, activation='relu')(latent_space)
decoded = layers.Dense(32, activation='relu')(decoded)
decoded = layers.Dense(64, activation='relu')(decoded)
output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)

# -model-
autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
learning_rate = 0.01  
optimizer = Adam(learning_rate=learning_rate)
autoencoder.compile(optimizer=optimizer, loss='mse')

# -training-
autoencoder.fit(X_train, X_train, epochs=200, batch_size=16, validation_data=(X_test, X_test), verbose=1)

# -encoder model to extract features-
encoder = models.Model(inputs=input_layer, outputs=latent_space)

# -extract encoded features-
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# -classifier using encoded features-
clf = RandomForestClassifier()
clf.fit(X_train_encoded, y_train)

# -evaluate classifier-
y_pred = clf.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy using Encoded Features: {accuracy:.2f}")

#----
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title("Confusion Matrix")
plt.show()
