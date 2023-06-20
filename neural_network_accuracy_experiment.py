import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
newsgroups_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
documents = newsgroups_data.data
labels = newsgroups_data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

# Define the hyperparameters to be tested
units = [16, 32, 64, 128]
epochs = [5, 10, 15, 20]
batch_sizes = [32, 64, 128, 256]

# Initialize lists to store the results
accuracy_results = []

# Perform the experiments
for unit in units:
    for epoch in epochs:
        for batch_size in batch_sizes:
            # Perform the training with specified hyperparameters
            vectorizer = TfidfVectorizer()
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            nn_model = Sequential([
                Dense(units=unit, activation='relu', input_dim=X_train_tfidf.shape[1]),
                Dense(units=1, activation='sigmoid')
            ])
            nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            nn_model.fit(X_train_tfidf.toarray(), y_train, epochs=epoch, batch_size=batch_size, verbose=0)
            nn_predictions = nn_model.predict(X_test_tfidf.toarray())
            nn_predictions = np.where(nn_predictions > 0.5, 1, 0)

            # Calculate the accuracy
            accuracy = accuracy_score(y_test, nn_predictions)

            # Store the results
            accuracy_results.append((unit, epoch, batch_size, accuracy))

# Create a DataFrame to store the results
df_results = pd.DataFrame(accuracy_results, columns=['Units', 'Epochs', 'Batch Size', 'Accuracy'])

# Save the results to a CSV file
df_results.to_csv('accuracy_results.csv', index=False)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.title("Neural Network Accuracy Enhancement Experiment")
plt.xlabel("Hyperparameters")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(rotation=45)

# Plotting the accuracy for different hyperparameter combinations
for i, unit in enumerate(units):
    for j, epoch in enumerate(epochs):
        for k, batch_size in enumerate(batch_sizes):
            accuracy = df_results.loc[
                (df_results['Units'] == unit) &
                (df_results['Epochs'] == epoch) &
                (df_results['Batch Size'] == batch_size)
            ]['Accuracy'].values[0]
            plt.scatter(f"({unit}, {epoch}, {batch_size})", accuracy)

plt.tight_layout()
plt.savefig('accuracy_results_plot.png')
plt.show()
