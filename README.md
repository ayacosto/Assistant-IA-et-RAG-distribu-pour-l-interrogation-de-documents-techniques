# Assistant IA et RAG distribué pour l'interrogation de documents techniques

**Auteures :** Aya Costo, Hafsa Zhari

Assistant RAG qui répond à des questions sur des documents PDF techniques (domaines Energy et Construction). Le chunking et les embeddings sont réalisés avec PySpark, la recherche avec FAISS, et la génération avec un LLM via Groq. L'interface est développée avec Gradio et FastAPI.

## Installation

```bash
pip install -r requirements.txt
```

## Clé API Groq

Crée un fichier `.env` à la racine du projet avec :

```
GROQ_API_KEY=ta_clé_groq
```

## Lancer l'application

```bash
python app_rag_assistant.py
```

Puis ouvre http://127.0.0.1:8000 dans ton navigateur. Au premier lancement, le modèle d'embedding (environ 440 Mo) est téléchargé automatiquement.

## Contenu du projet

- `app_rag_assistant.py` : interface Gradio et API FastAPI
- `spark_rag_pipeline.py` : pipeline PySpark (PDF, chunks, embeddings, index FAISS)
- `evaluate_rag.py` et `evaluation_metrics.py` : évaluation et calcul des métriques
- `generate_question.py` : génération d'une question de test à partir d'un chunk
- `data/vectordb/` : index FAISS et métadonnées déjà construits
- `data/evaluation_*` : résultats de l'évaluation