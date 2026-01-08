import pandas as pd
import yaml
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sentence_transformers import losses
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import re
import os

from common import BASE_PATH, MODEL_CONFIG_PATH, NLU_DATA_PATH, MODEL_SAVE_PATH

def load_model_config():
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_nlu_from_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    rows = []
    for item in data['nlu']:
        intent = item['intent']
        examples = item['examples'].strip().split('\n')
        for ex in examples:
            clean_text = re.sub(r'^-\s*', '', ex).strip().strip('"')
            if clean_text:
                rows.append({"intent": intent, "text": clean_text})
    return pd.DataFrame(rows)

def train():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    print(f"--- Szybki trening (Logistic Regression Head) dla: {model_name} ---")
    
    df = load_nlu_from_yaml(NLU_DATA_PATH)
    unique_intents = sorted(df['intent'].unique())
    label_map = {intent: i for i, intent in enumerate(unique_intents)}
    df['label'] = df['intent'].map(label_map)

    train_df, eval_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)

    model = SetFitModel.from_pretrained(
        model_name,
        labels=unique_intents,
        use_differentiable_head=False, 
        fix_mistral_regex=True
    )

    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        num_iterations=20,
        body_learning_rate=2e-5,
        end_to_end=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        column_mapping={"text": "text", "label": "label"} 
    )

    print("Rozpoczynam szybki trening kontrastowy...")
    trainer.train()

    print("\n--- Ewaluacja na zbiorze testowym ---")
    metrics = trainer.evaluate()
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")

    print("\n--- Szczegółowy raport klasyfikacji ---")
    y_true = eval_df['intent'].values
    y_pred = model.predict(eval_df['text'].tolist())
    
    print(classification_report(y_true, y_pred, zero_division=0))

    model.save_pretrained(MODEL_SAVE_PATH)
    print(f"--- Model zapisany w {MODEL_SAVE_PATH} ---")
