#weightfixgemma3bot.py


import re
import logging
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
import transformers.utils.logging as hf_logging

# -----------------------------
# Suppress Warnings
# -----------------------------
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()

# -----------------------------
# Load FAQ from file
# -----------------------------
def load_faq(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    qa_pairs = re.findall(r"Q:(.*?)\nA:(.*?)(?:\n\n|$)", content, re.S)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in qa_pairs]

# -----------------------------
# Embedding Model for Retrieval
# -----------------------------
print("Loading retrieval model (DistilBERT embeddings)...")
retriever = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

faq = load_faq("weight_faq.txt")
questions = [item["question"] for item in faq]
answers = [item["answer"] for item in faq]
question_embeddings = retriever.encode(questions, convert_to_tensor=False)

# -----------------------------
# Load Google Gemma 3 (1B IT)
# -----------------------------
print("Loading Google Gemma 3 model (this may take a moment)...")
model_name = "google/gemma-3-1b-it"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True  # CPU fallback if no CUDA
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=30,
    do_sample=False,  # Deterministic output
    temperature=0.0
)

# -----------------------------
# Greetings
# -----------------------------
greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}

# -----------------------------
# Answer Retrieval & Generation
# -----------------------------
def get_answer(user_question, threshold=0.5):
    uq_lower = user_question.lower().strip()

    # Handle greetings
    if any(uq_lower == g or uq_lower.startswith(g) for g in greetings):
        return "Hello! 👋 How can I assist you with the Weightfix program today?"

    # Retrieve closest FAQ entry
    user_emb = retriever.encode([user_question], convert_to_tensor=False)
    scores = cosine_similarity(user_emb, question_embeddings)[0]
    best_idx = scores.argmax()
    best_score = scores[best_idx]

    if best_score < threshold:
        return "Hmm 🤔 I’m not sure I understood that. Could you rephrase your question?"

    retrieved_q, retrieved_a = questions[best_idx], answers[best_idx]

    # Prompt LLM to tailor FAQ answer to user question
    prompt = (
        f"You are a helpful assistant for the Weightfix health program.\n"
        f"Here is a relevant FAQ entry:\n"
        f"Question: {retrieved_q}\n"
        f"Answer: {retrieved_a}\n\n"
        f"User Question: {user_question}\n"
        f"Respond clearly and concisely based on the FAQ context, but adapt it to the user's question:\n"
        f"Do not ask any leading question in your answer:\n"    
        f"Do not reference the FAQ document in your answer:\n"
        f"If unsure about answer to question, respond that 'you can call the customer service for more on this':\n"
    )

    raw_output = generator(prompt)[0]["generated_text"]
    completion = raw_output[len(prompt):].strip() if raw_output.startswith(prompt) else raw_output.strip()

    # Remove any trailing artifacts like extra prompts
    completion = re.split(r"(User Question:|FAQ Answer:)", completion)[0].strip()
    return completion

# -----------------------------
# Interactive Chatbot
# -----------------------------
def chatbot():
    print("🤖 Weightfix Bot: Hello! Ask me anything about the Weightfix program.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("🤖 Weightfix Bot: Thank you for chatting. Goodbye! 👋")
            break

        answer = get_answer(user_input)
        print(f"🤖 Weightfix Bot: {answer}\n")

if __name__ == "__main__":
    chatbot()


