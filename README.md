# Reward Model Training with TRL

This project trains a reward model using the `trl` library and the `trl-lib/ultrafeedback_binarized` dataset.

## Features

- Uses OpenAI GPT (`openai-gpt`) as the base model.
- Custom tokenizer and chat template integration.
- Tokenized prompt formatting for preference modeling.
- Custom loss logging with live plotting.
- Model training using `RewardTrainer` from the TRL library.