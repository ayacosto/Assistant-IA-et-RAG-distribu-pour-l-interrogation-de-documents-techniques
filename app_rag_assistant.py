import os
import faiss
import pandas as pd
import gradio as gr
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_LLM = "openai/gpt-oss-20b"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

index = faiss.read_index("data/vectordb/faiss.index")
metadata = pd.read_pickle("data/vectordb/metadata.pkl")
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
client = Groq(api_key=GROQ_API_KEY)


def retrieve(query, k=5):
    q_emb = embed_model.encode([QUERY_PREFIX + query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    results = metadata.iloc[indices[0]].copy()
    results["score"] = scores[0].astype(float)
    return results


def generate_answer(query, k=5):
    chunks = retrieve(query, k)
    context = "\n\n".join(
        f"[Source: {row['doc_id']}, domaine: {row['domain']}]\n{row['text']}"
        for _, row in chunks.iterrows()
    )
    prompt = f"""Réponds à la question en te basant uniquement sur le contexte fourni. Cite tes sources (doc_id).

Contexte:
{context}

Question: {query}

Réponse:"""
    response = client.chat.completions.create(
        model=MODEL_LLM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content, chunks


# ---------- API FastAPI ----------
app = FastAPI(title="Assistant RAG")


class Question(BaseModel):
    question: str
    k: int = 5


@app.post("/ask")
def ask(q: Question):
    answer, sources = generate_answer(q.question, q.k)
    return {
        "answer": answer,
        "sources": sources[["doc_id", "domain", "score"]].to_dict(orient="records"),
    }


# ---------- Interface Gradio ----------
def ask_ui(question, k):
    if not question.strip():
        return "Pose une question.", None
    answer, sources = generate_answer(question, int(k))
    return answer, sources[["doc_id", "domain", "score"]].round({"score": 3})


with gr.Blocks(title="Assistant RAG") as demo:
    gr.Markdown("# Assistant IA & RAG distribué\nPose une question technique (energy / construction).")
    question = gr.Textbox(label="Question", lines=2)
    k = gr.Slider(1, 10, value=5, step=1, label="Nombre de sources (k)")
    btn = gr.Button("Envoyer")
    answer = gr.Textbox(label="Réponse", lines=10)
    sources = gr.Dataframe(label="Sources")
    btn.click(ask_ui, inputs=[question, k], outputs=[answer, sources])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)