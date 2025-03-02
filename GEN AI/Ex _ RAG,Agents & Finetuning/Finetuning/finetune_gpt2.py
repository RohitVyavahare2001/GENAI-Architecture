from datasets import load_dataset
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Define data collator to handle variable-length inputs
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set to False because we're doing causal LM (GPT-2)
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=10
)

# Initialize Trainer with data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator
)

# Start fine-tuning
trainer.train()

# Save final model and tokenizer
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
