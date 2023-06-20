import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
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

# Vectorize the text data
print("Vectorizing text...")
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define the neural network model
units = 64
dropout_rate = 0.5
epochs = 15
batch_size = 128

print("Training Neural Network...")
nn_model = Sequential([
    Dense(units=units, activation='relu', input_dim=X_train_tfidf.shape[1]),
    Dropout(dropout_rate),
    Dense(units=units, activation='relu'),
    Dropout(dropout_rate),
    Dense(units=1, activation='sigmoid')
])
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train_tfidf.toarray(), y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Evaluate the neural network model
print("Evaluating Neural Network...")
nn_predictions = nn_model.predict_classes(X_test_tfidf.toarray())
print("Neural Network classification report:")
print(classification_report(y_test, nn_predictions))
