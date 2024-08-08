import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer

# Load the data
test_df = pd.read_csv('training_data/kmeans_regression/test_1.csv')

train_2 = pd.read_csv('training_data/kmeans_regression/train_2.csv')
train_3 = pd.read_csv('training_data/kmeans_regression/train_3.csv')
train_4 = pd.read_csv('training_data/kmeans_regression/train_4.csv')
train_5 = pd.read_csv('training_data/kmeans_regression/train_5.csv')
train_df = pd.concat([train_2, train_3, train_4, train_5], ignore_index=True)

# Print column names to ensure the correct name is used
print("Train dataset columns:", train_df.columns)
print("Test dataset columns:", test_df.columns)

# Convert pandas dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load the tokenizer and model
model_name = 'aaronfeller/PeptideCLM-23.4M-all'  # Replace with your model of choice
tokenizer = SMILES_SPE_Tokenizer('tokenizer/new_vocab.txt', 'tokenizer/new_splits.txt')
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Preprocess the datasets
# Replace 'SMILES' with the actual name of the text column and 'PAMPA' with the label column in your CSV files
def preprocess_function(examples):
    # Add the 'labels' field to the examples
    examples['labels'] = examples['PAMPA']
    tokenized_inputs = tokenizer(examples['SMILES'], padding='max_length', truncation=True, max_length=768)
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',           # Output directory
    num_train_epochs=10,              # Number of training epochs
    per_device_train_batch_size=16,   # Batch size for training
    per_device_eval_batch_size=16,    # Batch size for evaluation
    warmup_steps=0,                   # Number of warmup steps
    weight_decay=0.01,                # Strength of weight decay
    logging_dir='./logs',             # Directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",      # Evaluate every epoch
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()
