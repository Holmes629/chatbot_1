##############3 T5
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import pickle
import _1_Data_Extraction

current_directory = os.path.dirname(os.path.abspath(__file__))

# Disable WandB
os.environ["WANDB_DISABLED"] = "true"

# Extract text data
text_data = _1_Data_Extraction.Helper1()

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids,
        }

dataset = TextDataset(text_data, tokenizer)

model = T5ForConditionalGeneration.from_pretrained('t5-small')

output_dir = os.path.join(current_directory, "t5-text-generation")
model_dir = os.path.join(current_directory, "t5-text-generation-model")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=10,
    save_total_limit=2,
    max_steps=20,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Save additional data
max_sequence_len = 128  # Replace with actual max_length
max_seq_path = os.path.join(current_directory, "_2_max_seq_path.pkl")

with open(max_seq_path, 'wb') as handle:
    pickle.dump(max_sequence_len, handle)
