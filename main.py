# ==========================================
# 1. INSTALLATION & IMPORTS
# ==========================================
!pip install openai-whisper

import os
import pandas as pd
import numpy as np
import torch
import whisper
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "roberta-base"  # Excellent for syntax and semantic understanding
TRAIN_CSV = "/kaggle/input/shlhiring/dataset/csvs/train.csv"
TEST_CSV = "/kaggle/input/shlhiring/dataset/csvs/test.csv"
TRAIN_AUDIO_DIR = "/kaggle/input/shlhiring/dataset/audios/train/"
TEST_AUDIO_DIR = "/kaggle/input/shlhiring/dataset/audios/test/"

# ==========================================
# 2. PREPROCESSING: SPEECH-TO-TEXT (WHISPER)
# ==========================================
# We use Whisper to convert spoken audio into text.
stt_model = whisper.load_model("base").to(DEVICE)

def transcribe_data(df, audio_dir):
    """Transcribes audio files listed in a dataframe."""
    transcriptions = []
    for filename in tqdm(df['filename'], desc=f"Transcribing {audio_dir}"):
        # Append extension if missing
        file_path = os.path.join(audio_dir, filename if filename.endswith('.wav') else f"{filename}.wav")
        
        if not os.path.exists(file_path):
            transcriptions.append("")
            continue
            
        try:
            # Transcribe audio
            result = stt_model.transcribe(file_path)
            transcriptions.append(result['text'].strip())
        except Exception as e:
            print(f"Error on {filename}: {e}")
            transcriptions.append("")
            
    df['text'] = transcriptions
    return df[df['text'] != ""].reset_index(drop=True)

print("--- Step 1: Transcription ---")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df = transcribe_data(train_df, TRAIN_AUDIO_DIR)
test_df = transcribe_data(test_df, TEST_AUDIO_DIR)

# ==========================================
# 3. DATASET PREPARATION
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

class GrammarDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Split data for local validation
X_train, X_val, y_train, y_val = train_test_split(
    train_df['text'].tolist(), 
    train_df['label'].tolist(), 
    test_size=0.15, 
    random_state=42
)

train_set = GrammarDataset(tokenize_fn(X_train), y_train)
val_set = GrammarDataset(tokenize_fn(X_val), y_val)

# ==========================================
# 4. MODEL TRAINING (REGRESSION)
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    rmse = np.sqrt(mean_squared_error(labels, logits))
    return {"rmse": rmse}

# Load model with 1 output label for regression
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(DEVICE)

training_args = TrainingArguments(
    output_dir="./grammar_engine_results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="rmse",
    greater_is_better=False,
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\n--- Step 2: Training Regression Model ---")
trainer.train()

# ==========================================
# 5. FINAL EVALUATION (COMPULSORY RMSE)
# ==========================================
# Predict on the full training set
full_train_dataset = GrammarDataset(tokenize_fn(train_df['text'].tolist()))
train_preds = trainer.predict(full_train_dataset)
final_train_rmse = np.sqrt(mean_squared_error(train_df['label'], train_preds.predictions))

print("\n" + "="*30)
print(f"FINAL TRAINING RMSE: {final_train_rmse:.4f}")
print("="*30)

# ==========================================
# 6. INFERENCE & SUBMISSION
# ==========================================
print("\n--- Step 3: Generating Submission ---")
test_set = GrammarDataset(tokenize_fn(test_df['text'].tolist()))
test_preds = trainer.predict(test_set)

# Ensure scores are within valid 0-5 range
test_df['label'] = np.clip(test_preds.predictions.flatten(), 0.0, 5.0)

# Save only the required columns
test_df[['filename', 'label']].to_csv("submission.csv", index=False)
print("submission.csv created successfully.")
