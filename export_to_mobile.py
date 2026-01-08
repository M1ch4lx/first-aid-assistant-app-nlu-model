import os
import json
from common import BASE_PATH, NLU_DATA_PATH, FLOW_CONFIG_PATH, MODEL_SAVE_PATH
from nlu_train import load_nlu_from_yaml
import yaml
from run_bot import FirstAidBot
import shutil
from setfit.exporters.onnx import export_onnx

def export_project():
    print("--- Rozpoczynanie eksportu SetFit do ONNX ---")
    
    bot = FirstAidBot()
    if not bot.model:
        print("BŁĄD: Nie znaleziono wytrenowanego modelu! Uruchom 'train'.")
        return

    export_folder = os.path.join(BASE_PATH, "flutter_assets")
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    output_onnx_path = os.path.join(export_folder, "model.onnx")
    export_onnx(
        bot.model.model_body,
        bot.model.model_head,
        opset=18,
        output_path=output_onnx_path
    )
    print(f"Sukces! Pełny model wyeksportowany do: {output_onnx_path}")

    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
    for file_name in tokenizer_files:
        src = os.path.join(MODEL_SAVE_PATH, file_name)
        if os.path.exists(src):
            shutil.copy(src, export_folder)
            print(f"Skopiowano plik tokenizera: {file_name}")

    label_config = {i: intent for i, intent in enumerate(bot.model.labels)}
    with open(os.path.join(export_folder, "labels.json"), 'w', encoding='utf-8') as f:
        json.dump(label_config, f, ensure_ascii=False, indent=2)

    with open(FLOW_CONFIG_PATH, 'r', encoding='utf-8') as f:
        flow_data = yaml.safe_load(f)
    with open(os.path.join(export_folder, "app_config.yml"), 'w', encoding='utf-8') as f:
        yaml.dump(flow_data, f, allow_unicode=True, sort_keys=False)

    print("\n--- Eksport zakończony ---")