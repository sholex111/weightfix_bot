#weightfixbot.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==== FAQ DATA ====
faq = [
    {
        "question": "What is this program?",
        "leading_question": "Would you like me to tell you what the program is about?",
        "answer": "It's a comprehensive program designed to help you achieve a healthy weight."
    },
    {
        "question": "What is the main benefit of this program?",
        "leading_question": "Would you like me to tell you about the benefit of this program?",
        "answer": "It helps you achieve sustainable weight loss and a healthier lifestyle."
    },
    {
        "question": "What does the program include?",
        "leading_question": "Would you like me to tell you what the program includes?",
        "answer": "It includes personalized meal plans, exercise routines, and coaching."
    },
    {
        "question": "Who is this program for?",
        "leading_question": "Would you like me to tell you who this program is for?",
        "answer": "Anyone looking to lose weight and improve their health."
    },
    {
        "question": "How long does the program last?",
        "leading_question": "Would you like me to tell you how long the program lasts?",
        "answer": "Typically 12 weeks, but it can be customized."
    },
    {
        "question": "What is the time commitment required?",
        "leading_question": "Would you like me to tell you about the time commitment required?",
        "answer": "We recommend dedicating 3-5 hours per week to exercise and meal planning."
    },
    {
        "question": "How much does the program cost?",
        "leading_question": "Would you like me to tell you about the program's cost?",
        "answer": "The cost varies depending on the plan you choose. Please see our pricing page."
    },
    {
        "question": "Can I get a refund?",
        "leading_question": "Would you like me to tell you about our refund policy?",
        "answer": "Please refer to our refund policy on the website."
    },
    {
        "question": "Is a doctor's clearance needed?",
        "leading_question": "Would you like me to tell you if a doctor's clearance is needed?",
        "answer": "Yes, we strongly recommend it."
    },
    {
        "question": "Are there any forbidden foods?",
        "leading_question": "Would you like me to tell you if there are any forbidden foods?",
        "answer": "No, we focus on moderation and healthy choices."
    },
    {
        "question": "Will I need to count calories?",
        "leading_question": "Would you like me to tell you if you need to count calories?",
        "answer": "No, we provide pre-planned meals and guidance."
    },
    {
        "question": "Do I have to cook every day?",
        "leading_question": "Would you like me to tell you if you have to cook every day?",
        "answer": "No, we offer simple recipes and meal prep tips."
    },
    {
        "question": "Can I eat out at restaurants?",
        "leading_question": "Would you like me to tell you if you can eat out at restaurants?",
        "answer": "Yes, we provide strategies for dining out."
    },
    {
        "question": "What if I'm a vegetarian?",
        "leading_question": "Would you like me to tell you what happens if you're a vegetarian?",
        "answer": "We can customize the meal plans to fit your needs."
    },
    {
        "question": "Do I need a gym membership?",
        "leading_question": "Would you like me to tell you if you need a gym membership?",
        "answer": "No, many exercises can be done at home."
    },
    {
        "question": "How many times a week should I exercise?",
        "leading_question": "Would you like me to tell you how many times a week you should exercise?",
        "answer": "We recommend 3-5 times a week."
    },
    {
        "question": "What kind of exercises will I be doing?",
        "leading_question": "Would you like me to tell you what kind of exercises you will be doing?",
        "answer": "A mix of strength training and cardio."
    },
    {
        "question": "What is BMI?",
        "leading_question": "Would you like me to tell you what BMI is?",
        "answer": "BMI stands for Body Mass Index, a measure of body fat based on height and weight."
    },
    {
        "question": "How is BMI calculated?",
        "leading_question": "Would you like me to tell you how BMI is calculated?",
        "answer": "It's calculated by dividing your weight in kilograms by the square of your height in meters."
    },
    {
        "question": "What do my BMI results mean?",
        "leading_question": "Would you like me to tell you what your BMI results mean?",
        "answer": "BMI results indicate if you are underweight, a healthy weight, overweight, or obese."
    },
    {
        "question": "What kind of support do I get?",
        "leading_question": "Would you like me to tell you about the support you get?",
        "answer": "You will have access to a dedicated coach and a community forum."
    },
    {
        "question": "How quickly will I see results?",
        "leading_question": "Would you like me to tell you how quickly you will see results?",
        "answer": "Results vary, but most people see progress in the first few weeks."
    },
    {
        "question": "Will I lose weight fast?",
        "leading_question": "Would you like me to tell you if you will lose weight fast?",
        "answer": "No, our focus is on safe, sustainable weight loss of 1-2 pounds per week."
    },
    {
        "question": "What happens after the program ends?",
        "leading_question": "Would you like me to tell you what happens after the program ends?",
        "answer": "We provide resources to help you maintain your results."
    },
    {
        "question": "Is there an app for this program?",
        "leading_question": "Would you like me to tell you if there is an app for this program?",
        "answer": "Yes, we have a mobile app to track your progress and access resources."
    }
]

# ==== Vectorize Questions ====
questions = [item["question"] for item in faq]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# ==== Check User Said Yes ====
def is_yes(response):
    return response.strip().lower() in ["1", "yes", "y"]

# ==== Find Best Match ====
def find_best_match(user_input):
    user_vec = vectorizer.transform([user_input])
    scores = cosine_similarity(user_vec, question_vectors)
    best_idx = scores.argmax()
    return best_idx, scores[0][best_idx]

# ==== BMI Calculator ====
def calculate_bmi():
    try:
        weight = float(input("⚖️  Please enter your weight in kg: ").strip())
        height_cm = float(input("📏 Please enter your height in cm: ").strip())
        height_m = height_cm / 100
        bmi = weight / (height_m ** 2)
        print(f"\n📊 Your BMI is: {bmi:.1f}")
        if bmi < 18.5:
            print("Category: Underweight")
        elif 18.5 <= bmi < 25:
            print("Category: Normal weight")
        elif 25 <= bmi < 30:
            print("Category: Overweight")
        else:
            print("Category: Obesity")
    except ValueError:
        print("⚠️ Invalid input. Please enter numeric values for weight and height.")

# ==== Main Bot Logic ====
def faq_bot():
    print("🤖 Hello! I'm Chris, your weightfix bot.")
    print("I can help answer questions about the weight fix program.")
    print("Just type your question and I’ll respond. Type 'exit' to quit.")

    while True:
        user_input = input("\n❓ Your question: ").strip()
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break

        # Case 1: Direct BMI calculation request
        if "calculate" in user_input.lower() and "bmi" in user_input.lower():
            calculate_bmi()

        # Case 2: General FAQ flow
        else:
            idx, score = find_best_match(user_input)
            lead_q = faq[idx]["leading_question"]
            print(f"\n💬 {lead_q}")
            confirm = input("👉 Reply with 'yes' or '1' to proceed: ").strip()

            if is_yes(confirm):
                answer = faq[idx]['answer']
                print(f"\n✅ Answer: {answer}")

                # BMI mention in the answer
                if "bmi" in answer.lower():
                    bmi_ask = input("\n🧮 Would you like me to calculate your BMI? (yes/1): ").strip()
                    if is_yes(bmi_ask):
                        calculate_bmi()
            else:
                print("❌ Okay, feel free to ask in a different way.")

        # Ask if they want to continue — once only
        again = input("\n🤔 Do you have other questions? (yes/1 to continue, anything else to exit): ").strip()
        if not is_yes(again):
            print("👋 Alright, take care!")
            break

        

# ==== Run Bot ====
if __name__ == "__main__":
    faq_bot()
8



