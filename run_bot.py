import yaml
from setfit import SetFitModel

from common import MODEL_SAVE_PATH, FLOW_CONFIG_PATH

class FirstAidBot:
    def __init__(self, model_path=MODEL_SAVE_PATH, flow_path=FLOW_CONFIG_PATH):
        try:
            self.model = SetFitModel.from_pretrained(model_path, fix_mistral_regex=True)
            print(f"Wczytano wytrenowany model z {model_path}")
        except Exception:
            self.model = None
            print("Uwaga: Nie znaleziono wytrenowanego modelu. Użyj 'train'.")
        
        with open(flow_path, 'r', encoding='utf-8') as f:
            self.flow = yaml.safe_load(f)
            
        self.actions = {a['action']: a for a in self.flow['actions']}
        self.rules = self.flow['rules']

    def get_intent(self, text):
        if not self.model:
            return "MODEL_NOT_TRAINED", 0.0
        
        probs = self.model.predict_proba([text])[0]
        
        intent = self.model.predict([text])[0]
        
        score = float(probs.max()) 
        
        return intent, score

    def find_action(self, current_intent):
        for rule in self.rules:
            for step in rule['steps']:
                if 'intent' in step and step['intent'] == current_intent:
                    rule_steps = rule['steps']
                    idx = rule_steps.index(step)
                    if idx + 1 < len(rule_steps) and 'action' in rule_steps[idx+1]:
                        action_name = rule_steps[idx+1]['action']
                        return self.actions.get(action_name)
        return None


def run_bot():
    bot = FirstAidBot()
    print("Bot Pierwszej Pomocy uruchomiony. Wpisz zgłoszenie (np. 'Widzę kogoś nieprzytomnego').")
    while True:
        text = input("User > ")
        if text.lower() == 'exit': break
        
        intent, score = bot.get_intent(text)
        action = bot.find_action(intent)
        
        if action:
            print(f"Bot [{intent} {score:.2%}] > {action['message']}")
        else:
            print(f"Bot [{intent} {score:.2%}] > Nie rozumiem. Czy możesz opisać sytuację inaczej? Pamiętaj o bezpieczeństwie.")