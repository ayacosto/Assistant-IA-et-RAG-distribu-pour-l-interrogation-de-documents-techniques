import time
import pandas as pd
from app_rag_assistant import generate_answer, client, MODEL_LLM

df_q = pd.read_csv("data/evaluation_results.csv")[["question", "expected_doc_id"]]
results = []

for i, row in df_q.iterrows():
    question, expected_doc = row["question"], row["expected_doc_id"]
    print(f"Question {i+1}/{len(df_q)}...")

    start = time.time()
    answer, sources = generate_answer(question, k=5)
    latency = time.time() - start

    retrieved_docs = sources["doc_id"].tolist()
    hit = expected_doc in retrieved_docs
    rank = retrieved_docs.index(expected_doc) + 1 if hit else None

    time.sleep(8)

    context = "\n\n".join(sources["text"].tolist())
    judge_prompt = f"""Évalue sur une échelle de 1 à 5 si la réponse est pertinente ET fidèle au contexte fourni (juste le chiffre).

Question: {question}

Contexte:
{context}

Réponse: {answer}"""
    judge_score = client.chat.completions.create(
        model=MODEL_LLM,
        messages=[{"role": "user", "content": judge_prompt}],
    ).choices[0].message.content.strip()

    results.append({
        "question": question,
        "expected_doc_id": expected_doc,
        "hit@5": hit,
        "rank": rank,
        "latency_sec": round(latency, 2),
        "judge_score": judge_score,
    })
    print(f"  -> hit={hit}, rank={rank}, latence={latency:.2f}s, score={judge_score}")
    time.sleep(8)

pd.DataFrame(results).to_csv("data/evaluation_results_v2.csv", index=False)
print("Sauvegardé : data/evaluation_results_v2.csv")