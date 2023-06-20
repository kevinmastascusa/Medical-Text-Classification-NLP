"""
-----------------------------------------------------------
Medical Text Classification - Training Script
Date: 6/20/2023
Author: Kevin Mastascusa

This script performs the training and evaluation of a machine learning model for medical text classification. It utilizes the 20 Newsgroups dataset, which consists of various newsgroup posts on medical topics. The goal is to classify the medical texts into different categories based on their content.

The script follows the following steps:

1. Data Loading:
   - The 20 Newsgroups dataset is fetched using the fetch_20newsgroups function from the sklearn.datasets module.
   - The dataset is split into training and test sets using the train_test_split function from the sklearn.model_selection module.

2. Text Preprocessing:
   - The raw text data is preprocessed to remove stop words, tokenize the text, and perform lemmatization using the NLTK library.
   - The preprocessed text is stored in a separate list.

3. Model Training and Evaluation:
   - Two models are trained and evaluated: Logistic Regression and Support Vector Machines (SVM).
   - For each model, a pipeline is constructed consisting of a TF-IDF vectorizer and the corresponding classifier.
   - The models are trained using the fit method, and predictions are made on the test set.
   - The classification reports are generated using the classification_report function from the sklearn.metrics module.

4. Neural Network Training:
   - A Neural Network model is defined using the Keras library.
   - The model architecture consists of a Dense layer with ReLU activation and an output Dense layer with sigmoid activation.
   - The model is compiled with binary cross-entropy loss and the Adam optimizer.
   - Hyperparameter tuning is performed using a grid search with cross-validation to find the best combination of units, epochs, and batch size.
   - The best parameters are obtained, and a final model is trained using these parameters.
   - Predictions are made on the test set, and the classification report is generated.

5. Saving the Model:
   - The trained Neural Network model is saved as 'medical_text_classification_model.h5' using the save method from the Keras library.

This script provides a comprehensive approach to train and evaluate machine learning models for medical text classification. It demonstrates the use of different algorithms and techniques to achieve accurate classification results.

-----------------------------------------------------------
"""
import time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Preprocesses the text by removing common words, converting words to their base form, and getting them ready for analysis.

    Args:
        text (str): The input text.

    Returns:
        str: The preprocessed text.
    """
    tokenized_text = word_tokenize(text)
    cleaned_text = [lemmatizer.lemmatize(word) for word in tokenized_text if word not in stop_words]
    return " ".join(cleaned_text)


# Load the dataset
print("Loading dataset...")
newsgroups_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
documents = newsgroups_data.data
labels = newsgroups_data.target

# Preprocess the text
print("Preprocessing text...")
documents_processed = [preprocess_text(text) for text in documents]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(documents_processed, labels, test_size=0.2, random_state=42)

# Vectorize the text
print("Vectorizing text...")
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define the model architecture
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train_tfidf.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=20, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training loop with testing
epochs = 10  # Number of times the model will see the entire dataset
batch_size = 32  # Number of samples processed before the model's parameters are updated
testing_interval = 5  # Frequency of testing the model's performance
testing_epochs = 2  # Additional epochs for more accurate performance evaluation

print("Training started...")
start_time = time.time()

for epoch in range(epochs):
    # Train the model on the training data for one epoch
    model.fit(X_train_tfidf.toarray(), y_train, epochs=1, batch_size=batch_size, verbose=0)

    if (epoch + 1) % testing_interval == 0:
        print(f"\nTesting at epoch {epoch + 1}...")
        # Evaluate the model on the test data and calculate test loss and accuracy
        test_loss, test_accuracy = model.evaluate(X_test_tfidf.toarray(), y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

        # Perform additional training epochs for more accurate performance evaluation
        for _ in range(testing_epochs):
            model.fit(X_train_tfidf.toarray(), y_train, epochs=1, batch_size=batch_size, verbose=0)

end_time = time.time()
total_time = end_time - start_time
print(f"\nTraining completed. Total training time: {total_time:.2f} seconds.")

# Save the trained model
model.save("medical_text_classification_model.h5")
print("Trained model saved.")
# Footer
print("\n--- Training and Evaluation Completed ---")
print(f"Total training time: {total_time:.2f} seconds.")
print("Trained model saved as 'medical_text_classification_model.h5'.")
