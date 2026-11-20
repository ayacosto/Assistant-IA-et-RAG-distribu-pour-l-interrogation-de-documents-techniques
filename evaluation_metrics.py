import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/evaluation_results.csv")
df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
df["domain"] = df["expected_doc_id"].str.split("_").str[0]
n = len(df)

# --- Métriques ---
recall = df["hit@5"].mean()
hit1 = (df["rank"] == 1).mean()
mrr = (1 / df["rank"]).fillna(0).mean()

# Intervalle de confiance de Wilson (95 %) sur le Recall@5
z = 1.96
p = df["hit@5"].sum() / n
center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)

print(f"Questions : {n}")
print(f"Recall@5 : {recall:.1%}  (IC 95 % : {center-half:.1%} - {center+half:.1%})")
print(f"Hit@1 : {hit1:.1%}")
print(f"MRR : {mrr:.3f}")
print(f"Score LLM moyen : {df['judge_score'].mean():.2f}/5")
print(f"Latence : moy {df['latency_sec'].mean():.2f}s | méd {df['latency_sec'].median():.2f}s | max {df['latency_sec'].max():.2f}s")

by_domain = df.groupby("domain").agg(
    questions=("hit@5", "size"),
    recall5=("hit@5", "mean"),
    score_llm=("judge_score", "mean"),
    latence=("latency_sec", "mean"),
).round(3)
print("\nPar domaine :")
print(by_domain)

# --- Graphiques ---
fig, ax = plt.subplots(2, 2, figsize=(11, 7))

ranks = df["rank"].value_counts().reindex([1, 2, 3, 4, 5]).fillna(0)
labels = ["1", "2", "3", "4", "5", "absent"]
values = list(ranks.values) + [(~df["hit@5"]).sum()]
colors = ["#1a6f8a"] * 5 + ["#c0392b"]
ax[0, 0].bar(labels, values, color=colors)
ax[0, 0].set_title("Rang du document attendu")
ax[0, 0].set_ylabel("Nombre de questions")

ax[0, 1].bar(by_domain.index, by_domain["recall5"] * 100, color="#1a6f8a")
ax[0, 1].set_title("Recall@5 par domaine (%)")
ax[0, 1].set_ylim(0, 105)
for i, v in enumerate(by_domain["recall5"] * 100):
    ax[0, 1].text(i, v + 1, f"{v:.0f}%", ha="center")

sc = df["judge_score"].value_counts().reindex([1, 2, 3, 4, 5]).fillna(0)
ax[1, 0].bar(sc.index.astype(str), sc.values, color="#1a6f8a")
ax[1, 0].set_title("Distribution du score LLM")
ax[1, 0].set_xlabel("Score (1-5)")

bar_colors = ["#c0392b" if not h else "#1a6f8a" for h in df["hit@5"]]
ax[1, 1].bar(range(1, n + 1), df["latency_sec"], color=bar_colors)
ax[1, 1].axhline(df["latency_sec"].mean(), color="gray", ls="--", lw=1)
ax[1, 1].set_title("Latence par question (s), rouge = échec retrieval")
ax[1, 1].set_xlabel("Question")

for a in ax.flat:
    a.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("data/evaluation_charts.png", dpi=200, bbox_inches="tight")
plt.show()