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
    """Wczytuje konfigurację modelu i parametrów treningu z pliku YAML."""
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_nlu_from_yaml(file_path):
    """Wczytuje dane NLU i parsuje przykłady do formatu DataFrame."""
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
    # 1. Wczytanie konfiguracji z YAML
    config = load_model_config()
    
    model_cfg = config.get('model', {})
    train_cfg = config.get('training_arguments', {})
    
    model_name = model_cfg.get('base_model', "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    print(f"--- Wczytano konfigurację z: {MODEL_CONFIG_PATH} ---")
    print(f"--- Trening (Logistic Regression Head) dla modelu bazowego: {model_name} ---")
    
    # 2. Przygotowanie danych
    df = load_nlu_from_yaml(NLU_DATA_PATH)
    unique_intents = sorted(df['intent'].unique())
    label_map = {intent: i for i, intent in enumerate(unique_intents)}
    df['label'] = df['intent'].map(label_map)

    train_df, eval_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)

    # 3. Inicjalizacja modelu SetFit z nazwą z konfiguracji
    model = SetFitModel.from_pretrained(
        model_name,
        labels=unique_intents,
        use_differentiable_head=False, 
        fix_mistral_regex=True
    )

    # 4. Definicja argumentów treningowych z wartościami z YAML
    args = TrainingArguments(
        batch_size=train_cfg.get('batch_size', 16),
        num_epochs=train_cfg.get('num_epochs', 1),
        num_iterations=train_cfg.get('num_iterations', 20),
        body_learning_rate=float(train_cfg.get('body_learning_rate', 2e-5)),
        end_to_end=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    # 5. Inicjalizacja trenera
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        column_mapping={"text": "text", "label": "label"} 
    )

    print(f"Rozpoczynam trening (Epoki: {args.num_epochs}, Batch: {args.batch_size})...")
    trainer.train()

    # 6. Ewaluacja
    print("\n--- Ewaluacja na zbiorze testowym ---")
    metrics = trainer.evaluate()
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")

    print("\n--- Raport klasyfikacji ---")
    y_true = eval_df['intent'].values
    y_pred = model.predict(eval_df['text'].tolist())
    
    print(classification_report(y_true, y_pred, zero_division=0))

    # 7. Zapis modelu
    model.save_pretrained(MODEL_SAVE_PATH)
    print(f"--- Model zapisany w {MODEL_SAVE_PATH} ---")

if __name__ == "__main__":
    train()