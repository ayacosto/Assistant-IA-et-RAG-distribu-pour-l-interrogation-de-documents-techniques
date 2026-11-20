import os
import sys
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_LLM = "openai/gpt-oss-20b"

client = Groq(api_key=GROQ_API_KEY)
metadata = pd.read_pickle("data/vectordb/metadata.pkl")

# Optionnel : python generate_question.py energy   (ou construction)
domain = sys.argv[1] if len(sys.argv) > 1 else None
pool = metadata if domain is None else metadata[metadata["domain"] == domain]

# Tirer un chunk au hasard
chunk = pool.sample(1).iloc[0]

# Générer la question à partir de ce chunk
prompt = f"Génère UNE question précise à laquelle ce texte répond, en français. Donne uniquement la question.\n\n{chunk['text'][:1000]}"
question = client.chat.completions.create(
    model=MODEL_LLM,
    messages=[{"role": "user", "content": prompt}],
).choices[0].message.content.strip()

print("\nQUESTION GÉNÉRÉE :")
print(question)
print("\nCHUNK CORRESPONDANT :")
print(f"chunk_id : {chunk['chunk_id']} | doc_id : {chunk['doc_id']} | domaine : {chunk['domain']}")
print(chunk["text"][:800])