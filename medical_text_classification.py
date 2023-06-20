import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    tokenized_text = word_tokenize(text)
    cleaned_text = [lemmatizer.lemmatize(word) for word in tokenized_text if not word in stop_words]
    return " ".join(cleaned_text)


# Load the dataset
print("Loading dataset...")
newsgroups_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
documents = newsgroups_data.data
labels = newsgroups_data.target

# Preprocess the text
print("Preprocessing text...")
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

X_train_processed = [preprocess_text(i) for i in X_train]
X_test_processed = [preprocess_text(i) for i in X_test]

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train_processed)
X_test_tfidf = vectorizer.transform(X_test_processed)

# Logistic Regression
print("Training Logistic Regression model...")
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
lr_predictions = lr_model.predict(X_test_tfidf)
print("Logistic Regression classification report: ")
print(classification_report(y_test, lr_predictions))

# SVM
print("Training SVM model...")
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
svm_predictions = svm_model.predict(X_test_tfidf)
print("SVM classification report: ")
print(classification_report(y_test, svm_predictions))

# Neural Network with Keras
nn_model = Sequential()
nn_model.add(Dense(units=32, activation='relu', input_dim=X_train_tfidf.shape[1]))
nn_model.add(Dense(units=1, activation='sigmoid'))  # Binary classification
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train_tfidf.toarray(), y_train, epochs=10, verbose=1)
nn_probabilities = nn_model.predict(X_test_tfidf.toarray())
nn_predictions = (nn_probabilities > 0.5).astype(int)
print("Neural Network classification report: ")
print(classification_report(y_test, nn_predictions))

