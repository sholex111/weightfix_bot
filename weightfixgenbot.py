import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# -----------------------------
# Load FAQ
# -----------------------------
def load_faq(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    qa_pairs = re.findall(r"Q:(.*?)\nA:(.*?)(?:\n\n|$)", content, re.S)
    faq = [{"question": q.strip(), "answer": a.strip()} for q, a in qa_pairs]
    return faq

# -----------------------------
# Embedding model for retrieval
# -----------------------------
retriever = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
faq = load_faq("weight_faq.txt")
questions = [item["question"] for item in faq]
answers = [item["answer"] for item in faq]
question_embeddings = retriever.encode(questions)

# -----------------------------
# Generative model for tailored response
# -----------------------------
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# -----------------------------
# Greeting keywords
# -----------------------------
greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

def get_answer(user_question, threshold=0.4):
    # Check greetings first
    if user_question.lower() in greetings:
        return "Hello! 👋 How can I assist you with the Weightfix program today?"

    # Step 1: Retrieve closest FAQ
    user_emb = retriever.encode([user_question])
    scores = cosine_similarity(user_emb, question_embeddings)
    best_idx = scores.argmax()
    best_score = scores[0][best_idx]

    # Step 2: If score too low, ask to rephrase
    if best_score < threshold:
        return "Hmm 🤔 I’m not sure I understood that. Could you rephrase your question?"

    # Step 3: Generate tailored answer
    retrieved_q, retrieved_a = questions[best_idx], answers[best_idx]
    prompt = f"""You are an assistant answering based on an FAQ.

FAQ Question: {retrieved_q}
FAQ Answer: {retrieved_a}

User Question: {user_question}
Provide a helpful and natural response:"""

    response = generator(prompt, max_length=128, do_sample=False)[0]["generated_text"]
    return response

# -----------------------------
# Interactive chatbot
# -----------------------------
def chatbot():
    print("🤖 Weightfix Bot: Hello! Ask me anything about the Weightfix program.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("🤖 Weightfix Bot: Thank you for chatting. Goodbye! 👋")
            break

        answer = get_answer(user_input)
        print(f"🤖 Weightfix Bot: {answer}")
        print("🤖 Weightfix Bot: Can I help with anything else?")

if __name__ == "__main__":
    chatbot()
