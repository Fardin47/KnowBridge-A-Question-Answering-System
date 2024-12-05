import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

# 1. Load the dataset
dataset = load_dataset("christti/squad-augmented-v2")
small_dataset_train = dataset["train"].select(range(5000))  # Using only 5000 samples

# 2. Load the tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 3. Tokenize the dataset
def preprocess(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        truncation=True,
        max_length=384,
        stride=128,
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = inputs.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != 0:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 0:
                token_end_index -= 1
            if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_dataset_train = small_dataset_train.map(
    preprocess, batched=True, remove_columns=small_dataset_train.column_names
)
tokenized_dataset_validation = dataset["validation"].map(
    preprocess, batched=True, remove_columns=dataset["validation"].column_names
)

# 4. Prepare for training
data_collator = DefaultDataCollator()
training_args = TrainingArguments(
    output_dir="./models/fine_tuned_model",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    push_to_hub=False,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_validation,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 5. Train the model
trainer.train()

# 6. Save the fine-tuned model
model.save_pretrained("./models/fine_tuned_model")
tokenizer.save_pretrained("./models/fine_tuned_model")

print("Fine-tuned model saved in './models/fine_tuned_model'")
