# train_reward_model.py

import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainerCallback,
)
from trl import RewardTrainer, RewardConfig

# Constants
MODEL_NAME = "openai-gpt"
DATASET_NAME = "trl-lib/ultrafeedback_binarized"
MAX_LENGTH = 4096
OUTPUT_DIR = "./reward_model"
LOGGING_DIR = "./logs"

def extract_prompt(text):
    """Extract prompt from chosen sample."""
    return [{"role": "user", "content": text[0]["content"]}]

def prepare_dataset():
    """Load and preprocess the dataset."""
    dataset = load_dataset(DATASET_NAME, split="test")
    dataset = dataset.map(lambda sample: {**sample, "prompt": extract_prompt(sample["chosen"])})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if tokenizer.chat_template is None:
        tokenizer.chat_template = """{%- for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{%- endfor %}"""

    def tokenize(example):
        chat_text = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>"
            for message in example["prompt"]
        )
        return tokenizer(chat_text, truncation=True, padding="max_length", max_length=MAX_LENGTH)

    tokenized = dataset.map(tokenize)
    split = tokenized.train_test_split(test_size=0.1)
    return split["train"], split["test"], tokenizer

class LossLogger(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])

    def on_train_end(self, args, state, control, **kwargs):
        self.plot_loss()

    def plot_loss(self):
        if self.losses:
            plt.plot(self.losses)
            plt.xlabel("Steps")
            plt.ylabel("Training Loss")
            plt.title("Training Loss Over Steps")
            plt.show()

def main():
    train_data, eval_data, tokenizer = prepare_dataset()

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    config = RewardConfig(
        output_dir=OUTPUT_DIR,
        max_length=60,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        logging_dir=LOGGING_DIR,
        logging_steps=1,
    )

    trainer = RewardTrainer(
        model=model,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        callbacks=[LossLogger()],
    )

    trainer.train()

if __name__ == "__main__":
    main()
