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
    # 1. Załadowanie konfiguracji (zakładam istnienie Twojej funkcji load_model_config)
    # config = load_model_config()
    # model_name = config['model']['base_model']
    
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    print(f"--- Przygotowanie treningu (Nowy API Trainer) dla: {model_name} ---")
    
    # 2. Przygotowanie danych
    df = load_nlu_from_yaml(NLU_DATA_PATH)
    unique_intents = sorted(df['intent'].unique())
    label_map = {intent: i for i, intent in enumerate(unique_intents)}
    df['label'] = df['intent'].map(label_map)

    # Podział 80/20 ze stratyfikacją (zachowanie proporcji klas)
    train_df, eval_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)

    # 3. Inicjalizacja modelu z głowicą różniczkowalną (Torch)
    model = SetFitModel.from_pretrained(
        model_name,
        labels=unique_intents,
        use_differentiable_head=True,
        head_params={"out_features": len(unique_intents)},
        fix_mistral_regex=True
    )

    # 4. Konfiguracja argumentów treningowych (TrainingArguments)
    # Zwróć uwagę na krotki (faza_1, faza_2)
    args = TrainingArguments(
        batch_size=(16, 16),      # (embeddingi, głowica)
        num_epochs=(3, 10),      # (embeddingi, głowica) - głowica potrzebuje więcej epok!
        body_learning_rate=(2e-5, 1e-5), 
        head_learning_rate=1e-2,
        end_to_end=True,         # Pozwala na douczanie body podczas treningu głowicy
        l2_weight=0.01,
        evaluation_strategy="epoch", # Ewaluacja po każdej epoce
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    # 5. Inicjalizacja Trenera
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        column_mapping={"text": "text", "label": "label"} 
    )

    # 6. Trening
    print("Rozpoczynam dwufazowy trening...")
    trainer.train()

    # 7. Ewaluacja końcowa
    print("\n--- Ewaluacja na zbiorze testowym ---")
    metrics = trainer.evaluate()
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")

    # 8. Pełny raport klasyfikacji
    print("\n--- Szczegółowy raport klasyfikacji ---")
    y_true = eval_df['intent'].values
    y_pred = model.predict(eval_df['text'].tolist())
    
    # Używamy zero_division=0 aby uniknąć ostrzeżeń przy słabo wyuczonych klasach
    print(classification_report(y_true, y_pred, zero_division=0))

    # 9. Zapis modelu
    model.save_pretrained(MODEL_SAVE_PATH)
    print(f"--- Model zapisany w {MODEL_SAVE_PATH} ---")
