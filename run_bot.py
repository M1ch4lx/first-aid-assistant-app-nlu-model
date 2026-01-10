import yaml
from setfit import SetFitModel
from common import MODEL_SAVE_PATH, FLOW_CONFIG_PATH


class DialogueControl:
    def __init__(self, model_path=MODEL_SAVE_PATH, flow_path=FLOW_CONFIG_PATH, model=None):
        try:
            if model is not None:
                self.model = model
            else:
                self.model = SetFitModel.from_pretrained(model_path, fix_mistral_regex=True)
                print(f"Wczytano wytrenowany model z {model_path}")
        except Exception:
            self.model = None
            print("Nie znaleziono wytrenowanego modelu. Użyj 'train'.")
        
        with open(flow_path, 'r', encoding='utf-8') as f:
            self.flow = yaml.safe_load(f)
            
        self.actions = {a['action']: a for a in self.flow['actions']}
        self.rules = self.flow['rules']
        self.current_rule = None
        self.current_step = 0
        self.last_message = None

    def get_intent(self, text):
        if not self.model:
            return "MODEL_NOT_TRAINED", 0.0
        
        probs = self.model.predict_proba([text])[0]
        intent = self.model.predict([text])[0]
        score = float(probs.max())
        return intent, score

    def _find_rule_for_intent(self, intent):
        for rule in self.rules:
            for step in rule["steps"]:
                if "intent" in step and step["intent"] == intent:
                    return rule
                if "any" in step:
                    if intent in [c["intent"] for c in step["any"]]:
                        return rule
        return None

    def _find_rule_by_name(self, name):
        for rule in self.rules:
            if rule["rule"] == name:
                return rule
        return None

    def _next_action_in_rule(self, rule, current_index):
        steps = rule["steps"]
        actions = []
        goto_rule = None

        for i in range(current_index + 1, len(steps)):
            step = steps[i]
            if "action" in step:
                actions.append(step["action"])
            elif "goto" in step:
                goto_rule = step["goto"]
                break
            elif "intent" in step or "any" in step:
                break

        return actions, goto_rule

    def _format_action_output(self, action_data):
        if not action_data:
            return {"message": "(brak akcji)", "display": "", "special": False}
        
        special = "message" not in action_data

        message = action_data.get("message", action_data.get("action", ""))
        display = action_data.get("display", "")

        return {"message": message, "display": display, "special": special}

    def _execute_actions(self, actions):
        outputs = []
        for act_name in actions:
            act_data = self.actions.get(act_name)
            formatted = self._format_action_output(act_data)
            outputs.append(formatted)
            self.last_message = formatted["message"]
        return outputs 

    def _handle_goto(self, goto_rule_name):
        target_rule = self._find_rule_by_name(goto_rule_name)
        if not target_rule:
            return [{"message": f"(Błąd: nie znaleziono reguły '{goto_rule_name}')", "display": "", "special": False}]

        self.current_rule = target_rule
        self.current_step = 0

        steps = target_rule["steps"]
        first_intent_index = next((i for i, s in enumerate(steps) if "intent" in s or "any" in s), 0)

        output = []
        for i in range(first_intent_index):
            step = steps[i]
            if "action" in step:
                output.extend(self._execute_actions([step["action"]]))

        actions, next_goto = self._next_action_in_rule(target_rule, first_intent_index)
        if actions:
            output.extend(self._execute_actions(actions))

        if next_goto:
            output.extend(self._handle_goto(next_goto))

        return output

    def process_input(self, user_text):
        intent, score = self.get_intent(user_text)

        if intent == "REPEAT":
            return [{"message": self.last_message or "Nie mam nic do powtórzenia.", "display": "", "special": False}], intent, score

        if self.current_rule:
            steps = self.current_rule["steps"]
            for i in range(self.current_step + 1, len(steps)):
                step = steps[i]

                if "switch" in step:
                    for case in step["switch"]:
                        if case["case"] == intent:
                            output = self._handle_goto(case["goto"])
                            self.current_step = -1
                            return output, intent, score
                    continue

                matched = False
                if "intent" in step and step["intent"] == intent:
                    matched = True
                elif "any" in step and intent in [c["intent"] for c in step["any"]]:
                    matched = True

                if matched:
                    self.current_step = i
                    actions, goto_rule = self._next_action_in_rule(self.current_rule, i)
                    output = []
                    if actions:
                        output.extend(self._execute_actions(actions))
                    if goto_rule:
                        output.extend(self._handle_goto(goto_rule))
                    return output or [{"message": "Nie ma już dalszych kroków w tej procedurze", "display": "", "special": False}], intent, score

        rule = self._find_rule_for_intent(intent)
        if rule:
            self.current_rule = rule
            for i, step in enumerate(rule["steps"]):
                matched = False
                if "intent" in step and step["intent"] == intent:
                    matched = True
                elif "any" in step and intent in [c["intent"] for c in step["any"]]:
                    matched = True

                if matched:
                    self.current_step = i
                    actions, goto_rule = self._next_action_in_rule(rule, i)
                    output = []
                    if actions:
                        output.extend(self._execute_actions(actions))
                    if goto_rule:
                        output.extend(self._handle_goto(goto_rule))
                    return output or [{"message": "Nie ma już dalszych kroków w tej procedurze", "display": "", "special": False}], intent, score

        return [{"message": "Przepraszam, nie zrozumiałem. Możesz powtórzyć?", "display": "", "special": False}], intent, score


    def start_conversation(self):
        rule = self._find_rule_for_intent("START")
        if rule:
            self.current_rule = rule
            self.current_step = 0
            actions, goto_rule = self._next_action_in_rule(rule, 0)
            output = []
            if actions:
                output.extend(self._execute_actions(actions))
            if goto_rule:
                output.extend(self._handle_goto(goto_rule))
            return output
        return [{"message": "Nie udało się rozpocząć rozmowy.", "display": "", "special": False}]



def run_bot():
    bot = DialogueControl()
    print("Bot uruchomiony. Wpisz 'exit' aby zakończyć.\n")

    start_msg = bot.start_conversation()
    print(f"Bot > {start_msg}")

    while True:
        text = input("User > ").strip()
        if text.lower() == "exit":
            print("Bot > Zakończono rozmowę.")
            break

        reply, intent, score = bot.process_input(text)
        print(f"[Intent: {intent} ({score:.2f})]")
        print(f"Bot > {reply}")
