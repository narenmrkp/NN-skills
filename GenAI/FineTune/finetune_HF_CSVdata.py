import pandas as pd

data = {
    "text": [
        "I loved this product, it works great!",
        "The movie was terrible and too long.",
        "Excellent service and fast delivery.",
        "Not worth the money, very disappointed.",
        "Absolutely fantastic experience!",
        "I will never buy from them again.",
        "The food was delicious and the staff were friendly.",
        "Poor quality and bad customer support.",
        "Highly recommend this to everyone!",
        "It broke on the first day I used it."
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
df.to_csv("text_classification_sample.csv", index=False)

print("✅ Sample CSV saved as: text_classification_sample.csv")

# ✅ Step 1: Install dependencies
!pip install -q transformers datasets pandas scikit-learn gradio

# ✅ Step 2: Imports
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, TextClassificationPipeline
)
import gradio as gr

# ✅ Step 3: Load your CSV file (assumes already generated)
df = pd.read_csv("text_classification_sample.csv")

# ✅ Step 4: Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ✅ Step 5: Tokenization
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text", "__index_level_0__"])
test_dataset = test_dataset.remove_columns(["text", "__index_level_0__"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# ✅ Step 6: Load model and training args
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    logging_dir="./logs"
)

# ✅ Step 7: Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# ✅ Step 8: Create pipeline
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
label_map = {"LABEL_0": "Negative", "LABEL_1": "Positive"}

# ✅ Step 9: Define Gradio Interface
def classify_text(text):
    preds = pipe(text)[0]
    top = max(preds, key=lambda x: x['score'])
    label = label_map[top['label']]
    score = f"{top['score']*100:.2f}%"
    return f"Sentiment: {label} (Confidence: {score})"

# ✅ Step 10: Launch Gradio
gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=3, placeholder="Enter your review..."),
    outputs="text",
    title="📊 BERT Sentiment Classifier",
    description="Fine-tuned on your CSV data for binary classification (Positive/Negative)"
).launch()


