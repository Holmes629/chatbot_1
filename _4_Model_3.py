########### GPT-2
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
import _1_Data_Extraction

current_directory = os.path.dirname(os.path.abspath(__file__))

# Step 1: Extract text data
text_data = _1_Data_Extraction.Helper1()

# Step 2: Split the text into chunks for training
text_chunks = text_data.split('\n')[:500]  # Use a smaller subset for faster training

# Step 3: Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 uses EOS token for padding

# Tokenize and prepare the data
def tokenize_function(examples):
    return tokenizer(
        examples,
        truncation=True,
        padding=True,
        max_length=512,  # Adjust based on your data
        return_tensors='pt'
    )

# Step 4: Create a custom Dataset class
class TextDataset(Dataset):
    def __init__(self, text_chunks, tokenizer):
        self.tokenizer = tokenizer
        self.tokenized_data = self.tokenize_text(text_chunks)
    
    def tokenize_text(self, text_chunks):
        encodings = tokenizer(text_chunks, truncation=True, padding=True, max_length=512, return_tensors='pt')
        return encodings

    def __len__(self):
        return len(self.tokenized_data['input_ids'])
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_data.items()}
        item['labels'] = item['input_ids'].clone()  # Labels should be same as input_ids for language modeling
        return item

# Create the dataset
train_dataset = TextDataset(text_chunks, tokenizer)

# Create DataLoader (optional for more fine-grained control)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

# Step 5: Load GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Step 6: Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    save_steps=1000,
    save_total_limit=1,
    logging_dir='./logs',
    logging_steps=50,
    report_to=[],
    fp16=True
)

# Step 7: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Step 8: Train the model
print("Training the model...")
trainer.train()

# Step 9: Save the fine-tuned model and tokenizer
model_path = os.path.join(current_directory, "gpt2-text-generation-model")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
