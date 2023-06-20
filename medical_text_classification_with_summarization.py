import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BartTokenizer, BartForConditionalGeneration
import torch

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Set random seed for reproducibility
torch.manual_seed(42)

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download and load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Download and load the BART tokenizer and model for summarization
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Define the pre-processing functions
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

# Tokenize the input text
print("Tokenizing text...")
train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(X_test, truncation=True, padding=True, return_tensors='pt')

# Convert the labels to tensors
train_labels = torch.tensor(y_train, dtype=torch.long)
test_labels = torch.tensor(y_test, dtype=torch.long)

# Create the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=20)

# Set the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=1e-5)

# Move the model to the GPU
if device.type == 'cuda':
    model = model.to(device)

# Create the dataloaders
batch_size = 8  # Reduced batch size
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
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Define the accumulation steps
accumulation_steps = 4  # Number of steps to accumulate gradients before performing optimization

# Training loop
print("Training the BERT model...")
model.train()
total_loss = 0  # Initialize the total loss
for epoch in range(3):
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        # Move the input tensors to the GPU
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / accumulation_steps  # Divide the loss by the accumulation steps
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Perform optimization after accumulation steps
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
            total_loss = 0  # Reset the total loss

# Evaluation
print("Evaluating the BERT model...")
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_dataloader:
        # Move the input tensors to the GPU
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(logits.argmax(dim=-1).tolist())

# Print the classification report
target_names = newsgroups_data.target_names
print(classification_report(y_test, predictions, target_names=target_names))

# Text summarization
print("Performing text summarization...")
documents_summarized = []
for text in documents:
    inputs = bart_tokenizer.encode(text, truncation=True, max_length=1024, return_tensors='pt')
    inputs = inputs.to(device)
    summary_ids = bart_model.generate(inputs, num_beams=4, max_length=100, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    documents_summarized.append(summary)

# Print the summaries
for i, summary in enumerate(documents_summarized):
    print(f"Summary {i+1}: {summary}")
