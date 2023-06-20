import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, BartForConditionalGeneration, BartTokenizer
from transformers import AdamW
import torch

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Set random seed for reproducibility
torch.manual_seed(42)

# Download and load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Define the pre-processing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    tokenized_text = word_tokenize(text)
    cleaned_text = [lemmatizer.lemmatize(word) for word in tokenized_text if word not in stop_words]
    return " ".join(cleaned_text)


def summarize_text(text):
    inputs = bart_tokenizer.batch_encode_plus(
        [text],
        max_length=1024,
        min_length=50,
        truncation=True,
        padding='longest',
        return_tensors='pt'
    )

    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    return summary


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

# Tokenize the input text
print("Tokenizing text...")
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

# Convert the labels to tensors
train_labels = torch.tensor(y_train, dtype=torch.long)
test_labels = torch.tensor(y_test, dtype=torch.long)

# Create the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=20)

# Set the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=1e-5)

# Create the dataloaders
train_dataset = torch.utils.data.TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    train_labels
)
test_dataset = torch.utils.data.TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask'],
    test_labels
)

# Define the batch size
batch_size = 16

# Create the dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Training loop
print("Training the BERT model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()
for epoch in range(3):
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
print("Evaluating the BERT model...")
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(logits.argmax(dim=-1).tolist())

# Print the classification report
target_names = newsgroups_data.target_names
print(classification_report(y_test, predictions, target_names=target_names))

# Text summarization example
text = "I am experiencing symptoms such as coughing, fever, and shortness of breath. Is it possible that I have " \
       "COVID-19?"

summary = summarize_text(text)
print("Input Text:\n", text)
print("Summary:\n", summary)
