import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

# Load the dataset
newsgroups = fetch_20newsgroups(subset='all')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)

# Define the hyperparameters to be tested
units = [16, 32, 64, 128]
epochs = [5, 10, 15, 20]
batch_sizes = [32, 64, 128, 256]

# Initialize lists to store the results
speed_results = []

# Perform the experiments
for unit in units:
    for epoch in epochs:
        for batch_size in batch_sizes:
            # Start the timer
            start_time = time.time()

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

            # Calculate the elapsed time
            elapsed_time = time.time() - start_time

            # Store the results
            speed_results.append((unit, epoch, batch_size, elapsed_time))

# Create a DataFrame to store the results
df_results = pd.DataFrame(speed_results, columns=['Units', 'Epochs', 'Batch Size', 'Elapsed Time'])

# Save the results to a CSV file
df_results.to_csv('speed_results.csv', index=False)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.title("Neural Network Speed Enhancement Experiment")
plt.xlabel("Hyperparameters")
plt.ylabel("Elapsed Time (seconds)")
plt.grid(True)
plt.xticks(rotation=45)

# Plotting the elapsed time for different hyperparameter combinations
for i, unit in enumerate(units):
    for j, epoch in enumerate(epochs):
        for k, batch_size in enumerate(batch_sizes):
            elapsed_time = df_results.loc[
                (df_results['Units'] == unit) &
                (df_results['Epochs'] == epoch) &
                (df_results['Batch Size'] == batch_size)
            ]['Elapsed Time'].values[0]
            plt.scatter(f"({unit}, {epoch}, {batch_size})", elapsed_time)

plt.tight_layout()
plt.savefig('speed_results_plot.png')
plt.show()
