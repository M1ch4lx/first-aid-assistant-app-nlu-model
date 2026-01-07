import yaml
import os
from common import BASE_PATH, MODEL_CONFIG_PATH, NLU_DATA_PATH, FLOW_CONFIG_PATH

def init_project():
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        print(f"Utworzono folder: {BASE_PATH}")

    files = {
        MODEL_CONFIG_PATH: {
            "model": {
                "base_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            },
            "training_arguments": {
                "num_epochs": 1,
                "batch_size": 16,
                "use_amp": True,
                "num_iterations": 20,
                "learning_rate": 0.00002
            }
        },
        NLU_DATA_PATH: {
            "nlu": [
                {
                    "intent": "GREET",
                    "examples": "- \"Hello\"\n- \"Hi\"\n- \"How are you\"\n"
                }
            ]
        },
        FLOW_CONFIG_PATH: {
            "actions": [
                {
                    "action": "say_hello",
                    "message": "Hello",
                    "display": "Hello"
                }
            ],
            "rules": [
                {
                    "rule": "start",
                    "steps": [
                        {"intent": "START"},
                        {"action": "say_hello"}
                    ]
                },
                {
                    "rule": "greet",
                    "steps": [
                        {"intent": "GREET"},
                        {"action": "say_hello"}
                    ]
                }
            ]
        }
    }

    for path, content in files.items():
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(content, f, allow_unicode=True, sort_keys=False)
        print(f"Utworzono plik: {path}")

    print("\n--- Inicjalizacja zakończona sukcesem! ---")