import os
import re
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Step 1: Load FAQ from .txt file
# -----------------------------
def load_faq(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    qa_pairs = re.findall(r"Q:(.*?)\nA:(.*?)(?:\n\n|$)", content, re.S)
    faq = [{"question": q.strip(), "answer": a.strip()} for q, a in qa_pairs]
    return faq

# -----------------------------
# Step 2: Encode FAQ with DistilBERT
# -----------------------------
print("Loading model... (this may take a minute)")
model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

faq = load_faq("weight_faq.txt")
questions = [item["question"] for item in faq]
answers = [item["answer"] for item in faq]

question_embeddings = model.encode(questions, convert_to_tensor=True)

# -----------------------------
# Step 3: Search function
# -----------------------------
def find_best_answer(user_question):
    user_embedding = model.encode([user_question], convert_to_tensor=True)
    scores = cosine_similarity(
        user_embedding.cpu().numpy(), question_embeddings.cpu().numpy()
    )
    best_idx = scores.argmax()
    return answers[best_idx]

# -----------------------------
# Step 4: Interactive Chatbot
# -----------------------------
def chatbot():
    print("🤖 Weightfix Bot: Hello! Ask me anything about the Weightfix program.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("🤖 Weightfix Bot: Thank you for chatting. Goodbye! 👋")
            break

        answer = find_best_answer(user_input)
        print(f"🤖 Weightfix Bot: {answer}")
        print("🤖 Weightfix Bot: Can I help with anything else?")

if __name__ == "__main__":
    chatbot()
