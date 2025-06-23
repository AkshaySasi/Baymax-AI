from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# Load model and tokenizer
model_name = "facebook/opt-6.7b"  # Switched to an open-access model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

# Add pad token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Load and preprocess custom dataset
dataset = load_dataset("json", data_files="C:/Akshay/baymax.ai/backend/custom_emotional_data.json", split="train")
def preprocess_function(examples):
    inputs = [f"User: {dialog}\nBaymax:" for dialog in examples["input"]]
    outputs = [f" {resp}" for resp in examples["output"]]
    model_inputs = tokenizer(inputs, text_target=outputs, max_length=128, truncation=True, padding="max_length")
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./baymax_emotional_model_v3",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train
trainer.train()
trainer.save_model("./baymax_emotional_model_v3")
print("Training complete! Model saved to ./baymax_emotional_model_v3")