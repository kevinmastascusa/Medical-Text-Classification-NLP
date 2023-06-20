import nltk
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras import layers, utils
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokenized_text = word_tokenize(text)
    cleaned_text = [lemmatizer.lemmatize(word) for word in tokenized_text if not word in stop_words]
    return " ".join(cleaned_text)

print("Loading dataset...")
newsgroups_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
documents = newsgroups_data.data
labels = newsgroups_data.target

print("Preprocessing text...")
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

X_train_processed = [preprocess_text(i) for i in X_train]
X_test_processed = [preprocess_text(i) for i in X_test]

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train_processed)
X_test_tfidf = vectorizer.transform(X_test_processed)

print("Training Logistic Regression model...")
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
lr_predictions = lr_model.predict(X_test_tfidf)
print("Logistic Regression classification report: ")
print(classification_report(y_test, lr_predictions))

print("Training SVM model...")
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
svm_predictions = svm_model.predict(X_test_tfidf)
print("SVM classification report: ")
print(classification_report(y_test, svm_predictions))

print("Training Neural Network model...")
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

y_train_encoded = utils.to_categorical(y_train_encoded)
y_test_encoded = utils.to_categorical(y_test_encoded)

nn_model = tf.keras.Sequential()
nn_model.add(layers.Dense(units=32, activation='relu', input_dim=X_train_tfidf.shape[1]))
nn_model.add(layers.Dense(units=y_train_encoded.shape[1], activation='softmax'))
nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Use tf.data.Dataset to handle sparse matrix
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tfidf.toarray(), y_train_encoded))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_tfidf.toarray(), y_test_encoded))

nn_model.fit(train_dataset.batch(32), epochs=10)
nn_predictions = nn_model.predict(test_dataset.batch(32))

# Converting predictions to labels
nn_predictions_labels = tf.argmax(nn_predictions, axis=1).numpy()
print("Neural Network classification report: ")
print(classification_report(y_test_encoded, nn_predictions_labels))
