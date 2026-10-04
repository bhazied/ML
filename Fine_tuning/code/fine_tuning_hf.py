from dotenv import load_dotenv
from huggingface_hub import login
import os
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)
import sys

load_dotenv()  # Load environment variables from .env file
login() # Log in to Hugging Face Hub using the token from the .env file, aloso you can use the cli command `huggingface-cli login` to log in to the Hugging Face Hub using your token. This will store your credentials securely for future use.
print("Logged in to Hugging Face Hub successfully.")
def main():
    # 1. Load the dataset from the Hugging Face Hub
    print("Loading dataset...")
    # many dataset availble on huggingface, you can choose any dataset you want to use for finetuning
    #use stream to load the dataset in a streaming fashion, with this you can load large datasets without running out of memory, but you will not be able to shuffle the dataset, and you will not be able to use the dataset multiple times, 
    # get only 200 rows from the datasets to avoid running out of memery, dont forget is just an example!
    dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split="train[:200]") 
   
    # 2. Load the tokenizer and model from the Hugging Face Hub (ChatModel or causal LM model)
    # You can change this to any other model available on the Hugging Face Hub if you have a good memory use Qwen/Qwen3-0.6B
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"  
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Example of the dataset")
    raw_messages = dataset[0]["messages"]
    formatted_chat_string = tokenizer.apply_chat_template(raw_messages, tokenize=False)
    print(formatted_chat_string)
    print("---------------------------------------------\n")
    #sys.exit("data loaded and visualied")
    # 3. Tokenize the dataset
    def tokenize_function(examples):
        input_ids = []
        batch_attention_mask = []
        labels = []
        for messages in examples["messages"]:
            # Apply the chat template to the messages
            formatted_chat_string = tokenizer.apply_chat_template(messages, tokenize=False)
            # Tokenize the formatted chat string
            tokenized_output = tokenizer(formatted_chat_string, truncation=True, max_length=512)
            input_ids.append(tokenized_output["input_ids"])
            batch_attention_mask.append(tokenized_output["attention_mask"])
            labels.append(tokenized_output["input_ids"].copy())
        #return tokenizer(examples["messages"], truncation=True)
        return {"input_ids": input_ids, "attention_mask": batch_attention_mask, "labels" : labels}
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print("Example of the tokenized dataset")
    tokenized_messages = tokenized_datasets[0]
    print(tokenized_messages)
    print("---------------------------------------------\n")
    #sys.exit("data loaded and visualied")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_NAME)
# 4. Load the model for sequence classification
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # 5. Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",           #to calculate score at the end of every epoch and evaluate
        load_best_model_at_end=True,     # Revert to the best checkpoint when finished
        metric_for_best_model="loss",
        greater_is_better=False,
        learning_rate=2e-5,
        per_device_train_batch_size=1,  # train batch size to one raw at a time
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps to simulate a larger batch size
        per_device_eval_batch_size=1,   # 
        num_train_epochs=1,
        logging_steps=50,
        report_to="none",              # Prevents automatic tracking sync prompts (e.g., wandb)
        weight_decay=0.01,
        push_to_hub=True,  # Enable pushing to Hugging Face Hub
        hub_model_id="mrbhazied/tuned-SmolLM2-135M-Instruct-test-s",  # Change this to your desired model name on the Hub
    )

    # 6. Initialize the Trainer
    split_dataset = tokenized_datasets.train_test_split(test_size=0.1)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # 7. Train the model
    print("Starting training...")
    trainer.train()

    #9. save the tuned model
    print("Saving the tuned model locally by changing the output_dir in the training arguments.")
    trainer.save_model("./results")   #this will save the model in local folder
    tokenizer.save_pretrained("./results")  # Save the tokenizer as well
    print("saving the tuned model to the Hugging Face Hub...")
    trainer.push_to_hub()  # Push the model to the Hugging Face Hub

if __name__ == "__main__":
    main()

