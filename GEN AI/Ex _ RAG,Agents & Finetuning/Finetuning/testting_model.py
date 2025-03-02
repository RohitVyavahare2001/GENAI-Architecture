import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load fine-tuned GPT-2 model
model_path = "./final_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set padding token explicitly
tokenizer.pad_token = tokenizer.eos_token

def chat_with_model():
    print("Chat with your fine-tuned GPT-2! Type 'exit' to stop.")

    while True:
        # Get user input
        user_input = input("\nYou: ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        # Tokenize user input and create attention mask
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=1024)

        # Generate response (Explicitly pass `attention_mask`)
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # ✅ Fix: Explicitly passing attention mask
            max_length=150,  # Limit response length
            do_sample=True,  # Enable randomness for natural responses
            top_k=50,  # Sample from top 50 words
            top_p=0.95  # Use nucleus sampling
        )

        # Decode and print response
        response_text = tokenizer.decode(output[:, inputs["input_ids"].shape[-1]:][0], skip_special_tokens=True)
        print(f"GPT-2: {response_text}")

# Start chat
chat_with_model()
