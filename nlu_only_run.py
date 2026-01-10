from run_bot import DialogueControl

def run_nlu():
    bot = DialogueControl()
    print("Tryb NLU (wpisz 'exit' aby wyjść):")
    while True:
        text = input("User > ")
        if text.lower() == 'exit': break
        intent, score = bot.get_intent(text)
        print(f"Intent: {intent} ({score:.2%})")