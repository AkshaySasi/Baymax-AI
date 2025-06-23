import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import json

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_baymax_dataset():
    """Create a custom dataset with proper Baymax-style responses"""
    
    # Load EmpatheticDialogues
    try:
        dataset = load_dataset("empathetic_dialogues", trust_remote_code=True)
        print("Loaded EmpatheticDialogues dataset")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None
    
    # Create training examples with proper format
    training_examples = []
    
    # Use more data for better training
    train_data = dataset["train"].select(range(10000))  # Use 10k samples
    
    for example in train_data:
        # Get the context (what user said) and response
        user_input = example.get("context", "").strip()
        original_response = example.get("utterance", "").strip()
        emotion = example.get("emotion", "neutral").strip()
        
        if not user_input or not original_response:
            continue
        
        # Create Baymax-style response based on emotion
        baymax_response = create_baymax_response(original_response, emotion)
        
        # Format as conversation
        conversation = f"User: {user_input}\nBaymax: {baymax_response}<|endoftext|>"
        training_examples.append(conversation)
    
    return training_examples

def create_baymax_response(original_response, emotion):
    """Convert original response to Baymax-style response"""
    
    # Prefixes based on emotion
    prefixes = {
        "sadness": ["I can see you're hurting.", "I understand this is difficult.", "I'm here for you."],
        "anger": ["I can sense your frustration.", "It's okay to feel upset.", "Let's work through this together."],
        "fear": ["I understand you're worried.", "It's natural to feel anxious.", "You're safe here with me."],
        "joy": ["I'm so happy for you!", "That's wonderful to hear!", "Your joy is infectious!"],
        "surprise": ["That's quite unexpected!", "What an interesting development!", "Tell me more about that!"],
        "disgust": ["I can understand your concern.", "That does sound troubling.", "Let's talk about this."],
        "neutral": ["I'm listening.", "Tell me more.", "I'm here to help."]
    }
    
    import random
    emotion_lower = emotion.lower()
    
    # Get appropriate prefix
    if emotion_lower in prefixes:
        prefix = random.choice(prefixes[emotion_lower])
    else:
        prefix = random.choice(prefixes["neutral"])
    
    # Clean and adapt the original response
    response = original_response.strip()
    
    # Make it more empathetic and Baymax-like
    if len(response) > 100:
        response = response[:100] + "..."
    
    # Combine prefix with adapted response
    if response.lower().startswith(("i", "that", "it")):
        baymax_response = f"{prefix} {response}"
    else:
        baymax_response = f"{prefix} How can I support you with this?"
    
    return baymax_response

def fine_tune_model():
    """Fine-tune the model with improved settings"""
    
    # Create dataset
    training_examples = create_baymax_dataset()
    if not training_examples:
        print("Failed to create dataset")
        return
    
    print(f"Created {len(training_examples)} training examples")
    
    # Load model and tokenizer
    model_name = "microsoft/DialoGPT-small"  # Better for dialogue than GPT-2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = {"pad_token": "<pad>", "eos_token": "<|endoftext|>"}
    tokenizer.add_special_tokens(special_tokens)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # Tokenize training data
    def tokenize_function(examples):
        return tokenizer(
            examples,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
    
    # Create tokenized dataset
    tokenized_texts = []
    for example in training_examples:
        tokens = tokenizer(
            example,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        tokenized_texts.append({
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': tokens['input_ids'].squeeze()
        })
    
    # Split dataset
    split_idx = int(0.9 * len(tokenized_texts))
    train_dataset = tokenized_texts[:split_idx]
    eval_dataset = tokenized_texts[split_idx:]
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )
    
    # Training arguments with better settings
    training_args = TrainingArguments(
        output_dir="./baymax_emotional_model_v2",
        num_train_epochs=5,  # More epochs
        per_device_train_batch_size=2,  # Smaller batch size for stability
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 2*4 = 8
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb
        dataloader_pin_memory=False,
        learning_rate=5e-5,  # Lower learning rate for stability
    )
    
    # Custom dataset class
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = CustomDataset(train_dataset)
    eval_dataset = CustomDataset(eval_dataset)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model("./baymax_emotional_model_v2")
    tokenizer.save_pretrained("./baymax_emotional_model_v2")
    
    print("Training complete! Model saved to ./baymax_emotional_model_v2")
    
    # Test the model
    test_model()

def test_model():
    """Test the fine-tuned model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("./baymax_emotional_model_v2")
        model = AutoModelForCausalLM.from_pretrained("./baymax_emotional_model_v2")
        
        test_inputs = [
            "I'm feeling really sad today",
            "I'm so angry about this situation",
            "I'm worried about my future",
            "I'm really happy about my promotion"
        ]
        
        print("\n=== Testing Fine-tuned Model ===")
        for test_input in test_inputs:
            prompt = f"User: {test_input}\nBaymax:"
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Input: {test_input}")
            print(f"Response: {response[len(prompt):].strip()}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Testing failed: {e}")

if __name__ == "__main__":
    fine_tune_model()