# 🤖 Weightfix Chatbot Collection – From Simple to Advanced

A collection of chatbot implementations in Python, progressing from basic rule-based bots to advanced retrieval-augmented generation (RAG) systems powered by modern NLP and LLMs.

---

## ✨ Overview
This repository contains **five chatbot scripts**, each showcasing a different approach to conversational AI – starting with a simple command-line bot and evolving into sophisticated retrieval + generative models.

---

## 📂 Scripts Included

### 1. `simplechatbot.py`
A basic command-line chatbot named **Chris**.  
- Responds to greetings.  
- Provides date, time, and country (hardcoded: 🇬🇧 UK).  
- Can flip a yes/no or true/false response.  
- Generates random numbers.  

---

### 2. `weightfixbot.py`
An **FAQ chatbot for a weight loss program**, enhanced with NLP.  
- Uses **TF-IDF + Cosine Similarity** for semantic matching.  
- Answers common program questions without exact keyword matches.  
- Includes **BMI calculator** functionality.  

---

### 3. `weightfixretrievalbot_.py`
An **advanced retrieval chatbot** using **DistilBERT**.  
- Understands semantic meaning of user questions.  
- Reads FAQs from a text file for easy editing.  
- Provides more natural answers than keyword bots.  

---

### 4. `weightfixgenbot.py`
A **hybrid retrieval + generative chatbot**.  
- Uses **Sentence Transformers** for FAQ retrieval.  
- Refines answers with **FLAN-T5** generative model.  
- Produces **dynamic, personalized** responses.  
- Handles greetings + fallback prompts.  

---

### 5. `weightfixgemma3bot.py`
A **state-of-the-art RAG chatbot** using **Gemma 3 (1B IT)** + **DistilBERT**.  
- Retrieves FAQ answers via **cosine similarity**.  
- Synthesizes human-like responses with **Gemma LLM**.  
- Provides **accurate + conversational** flow.  
- Gracefully handles unanswerable questions.  

---

## 🚀 Installation & Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/sholex111/weightfix_bot.git
   cd your-repo-name
