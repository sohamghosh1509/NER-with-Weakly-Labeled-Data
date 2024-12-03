import torch
from transformers import (
    BertTokenizer,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    BertForMaskedLM,
    Trainer,
    TrainingArguments
)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 42  # Set your seed for reproducibility
setup_seed(SEED)

# Define training parameters directly in the script
pretrained_weights_path = 'bert-base-cased'  # Path to pre-trained BERT weights
unsupervised_train_data_path = '/kaggle/input/test-data/output_text (2).txt'  # Path to training data
do_lower_case = True  # Use lowercased text
max_len = 512  # Maximum token length
mlm_probability = 0.15  # Probability for masking tokens
lr = 5e-5  # Learning rate
epoch_num = 1  # Number of epochs
batch_size = 8  # Batch size

# Select the device (GPU if available)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_weights_path, do_lower_case=do_lower_case)

# Load the dataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=unsupervised_train_data_path,
    block_size=max_len
)

# Data collator for masked language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=mlm_probability
)

# Load the pre-trained BERT model for masked language modeling
mlm = BertForMaskedLM.from_pretrained(pretrained_weights_path).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    learning_rate=lr,
    num_train_epochs=epoch_num,
    per_device_train_batch_size=batch_size,
    save_strategy='epoch',
    seed=SEED
)

# Initialize the Trainer
trainer = Trainer(
    model=mlm,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save the model weights
mlm.save_pretrained('./pretrained_weights')
tokenizer.save_pretrained('./pretrained_weights')

import torch
from transformers import BertTokenizerFast, BertModel, Trainer, TrainingArguments
from torchcrf import CRF
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
# Custom Dataset class
class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) if key != 'word_ids' else val[idx]
            for key, val in self.encodings.items()
        }
        return item

def collate_fn(batch):
    keys = batch[0].keys()
    collated = {key: [] for key in keys}
    for b in batch:
        for key in keys:
            collated[key].append(b[key])
    # Stack tensors where applicable
    for key in ['input_ids', 'attention_mask', 'labels']:
        collated[key] = torch.stack(collated[key])
    return collated


def load_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r') as file:
        sentence = []
        label_seq = []
        for line in file:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label_seq)
                    sentence = []
                    label_seq = []
            else:
                parts = line.split()
                if len(parts) == 2:  # Ensure there are exactly two parts
                    word, label = parts
                    sentence.append(word)
                    label_seq.append(label)
                else:
                    continue
        if sentence:
            sentences.append(sentence)
            labels.append(label_seq)
    return sentences, labels


# Function to tokenize and align labels
def tokenize_and_align_labels(sentences, labels, label_map):
    tokenized_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_offsets_mapping=True,
    )
    all_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Assign the label for 'O' to special tokens
                label_ids.append(label_map['O'])
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[label[word_idx]])
            else:
                # For subword tokens, assign -100 so they are ignored in the loss computation
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

class BERT_CRF_NER(nn.Module):
    def __init__(self, num_labels, pretrained_weights_path):
        super(BERT_CRF_NER, self).__init__()
        # Load custom pretrained BERT weights
        self.bert = BertModel.from_pretrained(pretrained_weights_path, ignore_mismatched_sizes=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = label_map['O']
            mask = attention_mask.bool()
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)
            return emissions, predictions
        
from tqdm import tqdm
# Load custom pretrained tokenizer
tokenizer = BertTokenizerFast.from_pretrained('./pretrained_weights')

train_sentences, train_labels = load_data('/kaggle/input/bc5cdr-disease/train.txt')

unique_labels = set(label for doc in train_labels for label in doc)
label_list = sorted(unique_labels)
label_map = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label_map.items()}
num_labels = len(label_list)

if 'O' not in label_map:
    label_map['O'] = len(label_map)
    id2label[len(label_map) - 1] = 'O'
    num_labels += 1

# Tokenize and align labels
train_encodings = tokenize_and_align_labels(train_sentences, train_labels, label_map)

# Create datasets
train_dataset = NERDataset(train_encodings)

# Initialize model
model = BERT_CRF_NER(num_labels, './pretrained_weights')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training setup
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}')


import os

# Specify the directory to save the model
output_dir = './pretrained_weights_with_CRF'
os.makedirs(output_dir, exist_ok=True)

# Save the model's state_dict (parameters)
torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

# Save the tokenizer
tokenizer.save_pretrained(output_dir)
print(f"Model weights and tokenizer saved to {output_dir}")

def load_multiple_files(file_paths):
    sentences = []
    labels = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            sentence = []
            label_seq = []
            for line in file:
                line = line.strip()
                if line == "":
                    if sentence:
                        sentences.append(sentence)
                        labels.append(label_seq)
                        sentence = []
                        label_seq = []
                else:
                    parts = line.split()
                    if len(parts) == 2:  # Ensure there are exactly two parts
                        word, label = parts
                        sentence.append(word)
                        label_seq.append(label)
                    else:
                        continue
            if sentence:
                sentences.append(sentence)
                labels.append(label_seq)
    return sentences, labels

file_paths = ["/kaggle/input/weak-labelled-data/l1.txt", "/kaggle/input/weak-labelled-data/l2.txt", "/kaggle/input/weak-labelled-data/l3.txt", "/kaggle/input/weak-labelled-data/l4.txt"]

weak_train_sentences, weak_train_labels = load_multiple_files(file_paths)

weak_unique_labels = set(label for doc in weak_train_labels for label in doc)
weak_label_list = sorted(weak_unique_labels)
weak_label_map = {label: i for i, label in enumerate(weak_label_list)}
weak_id2label = {i: label for label, i in weak_label_map.items()}
weak_num_labels = len(weak_label_list)

if 'O' not in weak_label_map:
    weak_label_map['O'] = len(weak_label_map)
    weak_id2label[len(weak_label_map) - 1] = 'O'
    weak_num_labels += 1

weak_train_encodings = tokenize_and_align_labels(weak_train_sentences, weak_train_labels, weak_label_map)

weak_train_dataset = NERDataset(weak_train_encodings)

tokenizer = BertTokenizerFast.from_pretrained("./pretrained_weights_with_CRF")
weak_model = BERT_CRF_NER(weak_num_labels, "./pretrained_weights_with_CRF").to(device)

from torch.utils.data import DataLoader

def complete_labels(dataset, weak_model, tokenizer, label_map, device):
    model.eval()
    completed_labels = []

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        original_labels = batch['labels']
        
        with torch.no_grad():
            emissions, predictions = weak_model(input_ids=input_ids, attention_mask=attention_mask)
        
        for i, pred in enumerate(predictions):
            completed_label_seq = []
            for j, token_label in enumerate(original_labels[i]):
                if j < len(pred):
                    token_label = original_labels[i][j].item()
                    # Replace 'O' label in weak data with predicted label if not -100
                    if token_label == label_map['O'] and pred[j] != label_map['O'] and pred[j] != -100:
                        completed_label_seq.append(pred[j])
                    else:
                        completed_label_seq.append(token_label)
                else:
                    completed_label_seq.append(original_labels[i][j].item())
            completed_labels.append(completed_label_seq)
    
    return completed_labels

completed_labels = complete_labels(weak_train_dataset, weak_model, tokenizer, weak_label_map, device)

weak_train_encodings["labels"] = completed_labels

completed_weak_train_dataset = NERDataset(weak_train_encodings)

import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    confidences = [item['confidences'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    confidences_padded = pad_sequence(confidences, batch_first=True, padding_value=0.0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
        'confidences': confidences_padded
    }

from torch.utils.data import ConcatDataset

import torch
from transformers import AutoTokenizer, BertForTokenClassification
from torch.nn.functional import softmax
import numpy as np

tokenizer = BertTokenizerFast.from_pretrained("./pretrained_weights_with_CRF")

combined_sentences = train_sentences + weak_train_sentences
combined_labels = train_labels + weak_train_labels

combined_unique_labels = set(label for doc in combined_labels for label in doc)
combined_label_list = sorted(combined_unique_labels)
combined_label_map = {label: i for i, label in enumerate(combined_label_list)}
combined_id2label = {i: label for label, i in combined_label_map.items()}
combined_num_labels = len(combined_label_list)

model = BERT_CRF_NER(combined_num_labels, "./pretrained_weights_with_CRF").to(device)

if 'O' not in combinedlabel_map:
    combined_label_map['O'] = len(combined_label_map)
    combined_id2label[len(combined_label_map) - 1] = 'O'
    combined_num_labels += 1

combined_dataset = ConcatDataset([train_dataset, completed_weak_train_dataset])

import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to calculate confidence scores
def calculate_confidence_scores(logits):
    """
    Calculate confidence scores using the softmax probabilities of the logits.
    """
    probs = softmax(logits, dim=2)  # Convert logits to probabilities
    confidences = torch.max(probs, dim=2)[0].detach().cpu().numpy()  # Get max probability for each token
    return confidences  # Returns a NumPy array

class ConfidenceDatasetWrapper(Dataset):
    def __init__(self, dataset, model, device):
        self.dataset = dataset
        self.model = model
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        input_ids = torch.tensor(batch['input_ids']).to(self.device)
        attention_mask = torch.tensor(batch['attention_mask']).to(self.device)

        # Calculate logits and predictions
        with torch.no_grad():
            logits, predictions = self.model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            confidences = calculate_confidence_scores(logits)

        # Check the type of predictions and process accordingly
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.squeeze(0).cpu().numpy().tolist()  # Convert tensor to list
        elif isinstance(predictions, list):
            predictions = predictions  # Already a list, no conversion needed
        else:
            raise TypeError(f"Unexpected type for predictions: {type(predictions)}")

        # Add confidences and predictions to the batch
        batch['confidences'] = confidences.flatten().tolist()  # Convert NumPy array to list
        batch['predictions'] = predictions
        return batch

weak_train_dataset_with_confidences = ConfidenceDatasetWrapper(completed_weak_train_dataset, weak_model, device)

# DataLoader for confidence-enhanced dataset
weak_train_loader_with_confidences = DataLoader(
    weak_train_dataset_with_confidences, 
    batch_size=8, 
    shuffle=True, 
    collate_fn=collate_fn
)
    
# Process batches and calculate confidence
all_confidences = []
all_predictions = []
all_labels = []
for batch in tqdm(weak_train_loader_with_confidences):
    # Extract confidence scores, predictions, and labels
    confidences = batch['confidences']  # Confidence scores
    predictions = batch['predictions']  # Model's predictions
    labels = batch['labels'].flatten().tolist()

    all_confidences.extend(confidences)
    all_predictions.extend(predictions)
    all_labels.extend(labels)

def bin_confidences_and_calculate(bin_confidences, predicted_labels, strong_labels, num_bins=10):
    bin_confidences = np.array(bin_confidences).flatten()
    bins = np.linspace(3, 12, num_bins + 1)
    bin_accuracies = []
    bin_counts = []

    valid_length = min(len(bin_confidences), len(predicted_labels), len(strong_labels))
    bin_confidences = bin_confidences[:valid_length]
    predicted_labels = predicted_labels[:valid_length]
    strong_labels = strong_labels[:valid_length]

    for i in range(len(bins) - 1):
        bin_indices = [
            idx for idx, score in enumerate(bin_confidences)
            if bins[i] <= score < bins[i + 1]
        ]
        if bin_indices:
            bin_accuracy = accuracy_score(
                [predicted_labels[idx] for idx in bin_indices],
                [strong_labels[idx] for idx in bin_indices]
            )
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(len(bin_indices))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    return bins[:-1], bin_accuracies, bin_counts

bins, bin_accuracies, bin_counts = bin_confidences_and_calculate(all_confidences, all_predictions, all_labels)
smoothed_bin_accuracies = [min(0.95, acc) for acc in bin_accuracies]

# Calculate bin centers
bin_centers = (bins[:-1] + bins[1:]) / 2

# Plotting graph
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, smoothed_bin_accuracies, marker='o', label='Smoothed Bin Accuracies', color='blue')

# Adding labels and title
plt.title('CRF Score vs Smoothed Bin Accuracies')
plt.xlabel('CRF Score (Bin Centers)')
plt.ylabel('Smoothed Accuracy')
plt.ylim(0.7, 1.0)  # Adjust y-axis for better visualization
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Show the plot
plt.show()

model = BERT_CRF_NER(combined_num_labels, "./pretrained_weights_with_CRF").to(device)
train_dataset_with_confidences = ConfidenceDatasetWrapper(combined_dataset, model, tokenizer, device)
train_loader_with_confidences = DataLoader(train_dataset_with_confidences, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

def noise_aware_loss_function(emissions, labels, confidence_scores, crf_layer, attention_mask):

    # Clone and replace -100 in labels with label_map['O']
    labels = labels.clone()
    ignore_label = label_map['O']
    labels[labels == -100] = ignore_label
    
    mask = attention_mask.bool()
    
    nll_loss = -crf_layer(emissions, labels, mask=mask, reduction='none')
    
    # Compute per-instance confidence by averaging over valid tokens
    valid_confidences = confidence_scores * mask
    sum_confidences = valid_confidences.sum(dim=1)
    num_valid_tokens = mask.sum(dim=1) + 1e-8
    per_instance_confidence = sum_confidences / num_valid_tokens
    
    noise_aware_loss = (per_instance_confidence * nll_loss).mean()
    return noise_aware_loss
    

model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 1

for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(train_loader_with_confidences):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        weak_labels = batch["labels"].to(device)
        confidence_scores = batch["confidences"].to(device)
        
        emissions = model.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = model.classifier(emissions)

        loss = noise_aware_loss_function(emissions, weak_labels, confidence_scores, model.crf, attention_mask)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader_with_confidences)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}')


output_dir = './noise_aware_pretrained_weights_with_CRF'
os.makedirs(output_dir, exist_ok=True)

torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

tokenizer.save_pretrained(output_dir)
print(f"Model weights and tokenizer saved to {output_dir}")

from seqeval.metrics import classification_report, f1_score
   
# Load data
train_sentences, train_labels = load_data('/kaggle/input/bc5cdr-disease/train.txt')
dev_sentences, dev_labels = load_data('/kaggle/input/bc5cdr-disease/dev.txt')
test_sentences, test_labels = load_data('/kaggle/input/bc5cdr-disease/test.txt')

# Create label mappings
unique_labels = set(label for doc in train_labels for label in doc)
label_list = sorted(unique_labels)
label_map = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label_map.items()}
num_labels = len(label_list)

# Ensure 'O' label exists in label_map
if 'O' not in label_map:
    label_map['O'] = len(label_map)
    id2label[len(label_map) - 1] = 'O'
    num_labels += 1

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained('/kaggle/working/noise_aware_pretrained_weights_with_CRF')

# Tokenize and align labels
train_encodings = tokenize_and_align_labels(train_sentences, train_labels, label_map)
test_encodings = tokenize_and_align_labels(test_sentences, test_labels, label_map)

# Create datasets
train_dataset = NERDataset(train_encodings)
test_dataset = NERDataset(test_encodings)

# Initialize model
model = BERT_CRF_NER(num_labels, '/kaggle/working/noise_aware_pretrained_weights_with_CRF')

# Training setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 5

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

# Create DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Function to evaluate model
def evaluate(model, dataloader, label_map, id2label):
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get predictions
            _, preds = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Convert predictions and labels to lists
            preds = [[id2label[label] for label in pred] for pred in preds]
            labels = [[id2label[label.item()] if label.item() != -100 else 'O' for label in sent] for sent in labels]
            
            # Remove padding and subword labels
            for i in range(len(labels)):
                true_sent, pred_sent = [], []
                for j in range(len(labels[i])):
                    if labels[i][j] != 'O' or attention_mask[i][j].item() == 1:
                        true_sent.append(labels[i][j])
                        pred_sent.append(preds[i][j])
                true_labels.append(true_sent)
                predictions.append(pred_sent)
    
    # Calculate metrics
    report = classification_report(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return report, f1

# Run evaluation
report, f1 = evaluate(model, test_loader, label_map, id2label)
print("Classification Report:\n", report)
print("F1 Score:", f1)




