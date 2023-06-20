import nltk
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [lemmatizer.lemmatize(word) for word in tokenized_text if word not in stop_words]
    return " ".join(cleaned_text)

print("Loading dataset...")
newsgroups_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
documents = newsgroups_data.data
labels = newsgroups_data.target

print("Preprocessing text...")
documents_processed = [preprocess_text(text) for text in documents]

X_train, X_test, y_train, y_test = train_test_split(documents_processed, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

num_classes = len(newsgroups_data.target_names)
y_train_categorical = to_categorical(y_train, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

lr_pipeline = Pipeline([
    ('lr', LogisticRegression(max_iter=1000))
])
print("Training and evaluating Logistic Regression...")
lr_pipeline.fit(X_train_transformed, y_train)
lr_predictions = lr_pipeline.predict(X_test_transformed)
print("Logistic Regression classification report:")
print(classification_report(y_test, lr_predictions))

svm_pipeline = Pipeline([
    ('svm', SVC())
])
print("Training and evaluating SVM...")
svm_pipeline.fit(X_train_transformed, y_train)
svm_predictions = svm_pipeline.predict(X_test_transformed)
print("SVM classification report:")
print(classification_report(y_test, svm_predictions))

def create_nn_model(units=32):
    input_dim = X_train_transformed.shape[1]
    nn_model = Sequential()
    nn_model.add(Dense(units=units, activation='relu', input_dim=input_dim))
    nn_model.add(Dense(units=num_classes, activation='softmax'))
    nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return nn_model

param_grid = {
    'units': [16, 32, 64],
    'epochs': [5, 10, 15],
    'batch_size': [32, 64, 128]
}

# Convert sparse matrix to dense
X_train_transformed_dense = X_train_transformed.toarray()

grid_search = GridSearchCV(estimator=KerasClassifier(build_fn=create_nn_model), param_grid=param_grid, cv=3)
grid_search.fit(X_train_transformed_dense, y_train_categorical)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters: ", best_params)
print("Best Score: ", best_score)
