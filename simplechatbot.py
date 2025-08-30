import random
from datetime import datetime

class SimpleBot:
    def __init__(self, name):
        self.name = name
        self.text = ""

    def get_input(self):
        self.text = input("You --> ").strip().lower()

    def respond(self):
        txt = self.text

        # Exit condition
        if txt in ["exit", "quit", "goodbye"]:
            print("🤖 Goodbye! Have a nice day!")
            return False

        # Supported simple commands
        if txt in ["hello", "hi", "hey"]:
            print("🤖 Hello! How can I help you?")
        elif "date" in txt:
            today = datetime.now().strftime("%B %d, %Y")
            print(f"🤖 Today's date is {today}.")
        elif "time" in txt:
            now = datetime.now().strftime("%H:%M")
            print(f"🤖 The current time is {now}.")
        elif "country" in txt:
            print("🤖 You are in the United Kingdom.")
        elif "yes" in txt and "no" in txt:
            choice = random.choice(["Yes", "No"])
            print(f"🤖 I would say {choice}, if you don't mind.")
        elif "true" in txt and "false" in txt:
            choice = random.choice(["True", "False"])
            print(f"🤖 I believe the answer is {choice}.")
        elif "random number" in txt or "pick a number" in txt:
            number = random.randint(1, 100)
            print(f"🤖 Here is a random number for you: {number}.")
        elif "thank" in txt:
            print("🤖 You're welcome!")
        else:
            # Anything else is too complex
            print(
                "🤖 I do not understand, please ask me simple questions like:\n"
                "hello; today's date; which country am I; say yes or no; true or false; pick a random number"
            )
        return True


# ====== MAIN LOOP ======
if __name__ == "__main__":
    bot = SimpleBot(name="Chris")
    print("🤖 Hello! I am Chris.How may I help you? \nType 'exit' to quit.")

    while True:
        bot.get_input()
        if not bot.respond():
            break
