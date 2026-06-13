# ==============================================
# 06_xlm_roberta_model.py
# XLM-RoBERTa fine-tuning with leakage-free
# group-aware split, weighted loss and dropout
# Run AFTER 01_data_preparation.py
# ==============================================

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datasets import Dataset
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AutoConfig,
                          TrainingArguments, Trainer)
import evaluate

from config import (
    MASTER_CSV, SAVED_MODELS_FOLDER,
    RANDOM_SEED, XLM_MODEL_ID, XLM_MAX_LEN,
    XLM_BATCH_SIZE, XLM_EPOCHS, XLM_LR,
    XLM_WEIGHT_DECAY
)

os.makedirs(SAVED_MODELS_FOLDER, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(SAVED_MODELS_FOLDER, "xlm_roberta_scam")

# ==============================================
# STEP 1: LEAKAGE-FREE SEGMENTATION
# Split at transcript level BEFORE segmenting
# ==============================================

def segment_text(text, tokenizer, max_len=50, overlap=10):
    if not isinstance(text, str) or not text.strip():
        return []

    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_len:
        return [tokenizer.decode(tokens)]

    segments = []
    step     = max_len - overlap
    for i in range(0, len(tokens), step):
        chunk = tokenizer.decode(tokens[i: i + max_len])
        segments.append(chunk)
        if i + max_len >= len(tokens):
            break
    return segments


def prepare_xlm_data(tokenizer):
    df = pd.read_csv(MASTER_CSV)

    print(f"Loaded {len(df)} transcripts")
    print(f"Scam: {(df['label']==1).sum()} | Normal: {(df['label']==0).sum()}\n")

    # Transcript-level leakage-free split
    gss = GroupShuffleSplit(
        n_splits     = 1,
        test_size    = 0.2,
        random_state = RANDOM_SEED
    )
    train_idx, val_idx = next(
        gss.split(df, groups=df['source_file'])
    )
    train_raw = df.iloc[train_idx]
    val_raw   = df.iloc[val_idx]

    print(f"Train transcripts : {len(train_raw)} | Val transcripts: {len(val_raw)}")

    # Segment AFTER split to prevent leakage
    def segment_df(df_input):
        rows = []
        for _, row in df_input.iterrows():
            chunks = segment_text(row['content'], tokenizer)
            for chunk in chunks:
                rows.append({
                    'text'       : chunk,
                    'label'      : int(row['label']),
                    'source_file': row['source_file']
                })
        return pd.DataFrame(rows)

    train_df = segment_df(train_raw)
    val_df   = segment_df(val_raw)

    print(f"\nTrain segments : {len(train_df)} | Val segments: {len(val_df)}")
    print(f"Train — Scam: {(train_df['label']==1).sum()} | Normal: {(train_df['label']==0).sum()}")
    print(f"Val   — Scam: {(val_df['label']==1).sum()}   | Normal: {(val_df['label']==0).sum()}")

    return train_df, val_df


# ==============================================
# STEP 2: WEIGHTED LOSS TRAINER
# ==============================================

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ==============================================
# STEP 3: TRAIN
# ==============================================

def train_xlm_model(train_df, val_df, tokenizer):

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding    = "max_length",
            truncation = True,
            max_length = XLM_MAX_LEN
        )

    train_ds = Dataset.from_pandas(
        train_df[['text', 'label']].reset_index(drop=True)
    ).map(tokenize, batched=True)

    val_ds = Dataset.from_pandas(
        val_df[['text', 'label']].reset_index(drop=True)
    ).map(tokenize, batched=True)

    train_ds.set_format("torch",
                        columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch",
                      columns=["input_ids", "attention_mask", "label"])

    # Class weights
    labels        = train_df['label'].values
    weights       = compute_class_weight('balanced',
                                          classes=np.unique(labels),
                                          y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"\nClass weights: Normal={class_weights[0]:.3f}, Scam={class_weights[1]:.3f}")

    # Load config with increased dropout
    model_config = AutoConfig.from_pretrained(
        XLM_MODEL_ID,
        num_labels                   = 2,
    )

    # Load model with updated config
    model = AutoModelForSequenceClassification.from_pretrained(
        XLM_MODEL_ID,
        config = model_config
    )

    # Metrics
    metric_acc = evaluate.load("accuracy")
    metric_f1  = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_acc.compute(
                predictions=preds, references=labels)['accuracy'],
            "f1": metric_f1.compute(
                predictions=preds, references=labels,
                average='weighted')['f1']
        }

    training_args = TrainingArguments(
        output_dir                  = "./xlm_temp",
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        logging_strategy            = "epoch",
        learning_rate               = XLM_LR,
        per_device_train_batch_size = XLM_BATCH_SIZE,
        per_device_eval_batch_size  = XLM_BATCH_SIZE,
        num_train_epochs            = XLM_EPOCHS,
        weight_decay                = XLM_WEIGHT_DECAY,
        warmup_ratio                = 0.1,
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        report_to                   = "none"
    )

    trainer = WeightedTrainer(
        class_weights   = class_weights,
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        compute_metrics = compute_metrics
    )

    print("\n🚀 Training XLM-RoBERTa...")
    trainer.train()

    return trainer, val_ds, val_df


# ==============================================
# STEP 4: EVALUATE + PLOTS
# ==============================================

def evaluate_xlm(trainer, val_ds, val_df):
    preds_output = trainer.predict(val_ds)
    y_true = preds_output.label_ids
    y_pred = np.argmax(preds_output.predictions, axis=1)

    print("\n" + "="*50)
    print("📋 CLASSIFICATION REPORT — XLM-RoBERTa")
    print("="*50)
    print(classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Scam'],
        digits=4
    ))


    # ← ADD HERE, still inside evaluate_xlm()
    report = classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Scam'],
        digits=4
    )
    report_path = os.path.join(SAVED_MODELS_FOLDER, 'xlm_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}\n\n")
        f.write(report)
    print(f"✅ Classification report saved: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Scam'],
                yticklabels=['Normal', 'Scam'])
    plt.title('Confusion Matrix — XLM-RoBERTa')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVED_MODELS_FOLDER, 'xlm_confusion_matrix.png'),
                dpi=150)
    plt.show()

    # Loss curve
    history    = trainer.state.log_history
    log_df     = pd.DataFrame(history)
    train_loss = log_df[log_df['loss'].notna()][['epoch', 'loss']]
    val_loss   = log_df[log_df['eval_loss'].notna()][['epoch', 'eval_loss']]

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss['epoch'], train_loss['loss'],
             'b-o', label='Training Loss')
    plt.plot(val_loss['epoch'],   val_loss['eval_loss'],
             'r-s', label='Validation Loss')
    plt.title('Training vs Validation Loss — XLM-RoBERTa')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVED_MODELS_FOLDER, 'xlm_loss_curve.png'), dpi=150)
    plt.show()
    print("✅ Plots saved")


# ==============================================
# MAIN
# ==============================================

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(XLM_MODEL_ID)

    print("="*50)
    print("STEP 1: Preparing segments (leakage-free)")
    print("="*50)
    train_df, val_df = prepare_xlm_data(tokenizer)

    print("\n" + "="*50)
    print("STEP 2: Training XLM-RoBERTa")
    print("="*50)
    trainer, val_ds, val_df = train_xlm_model(train_df, val_df, tokenizer)

    print("\n" + "="*50)
    print("STEP 3: Evaluating")
    print("="*50)
    evaluate_xlm(trainer, val_ds, val_df)

    # Save
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
    print("➡️  Next step: run 05_shap_xlm.py")

