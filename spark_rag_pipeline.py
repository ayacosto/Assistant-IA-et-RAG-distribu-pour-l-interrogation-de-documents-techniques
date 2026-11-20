"""
spark_rag_pipeline.py
Pipeline RAG : PDF -> chunks (Spark) -> embeddings -> index FAISS
"""
import os
import sys
import shutil
import tarfile
import traceback
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import faiss
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

DOMAINS = ["energy", "construction"]
MAX_PDFS_PER_DOMAIN = 50
CHUNK_SIZE, CHUNK_OVERLAP = 1600, 200
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

RAW_DIR = Path("data/raw_pdfs")
CHUNKS_DIR = "data/chunks/chunks"
EMB_DIR = "data/embeddings/chunks_with_embeddings"
VDB_DIR = Path("data/vectordb")


# ---------- 1. Téléchargement des PDF ----------
def download_pdfs():
    for domain in DOMAINS:
        target = RAW_DIR / domain
        if target.exists() and any(target.glob("*.pdf")):
            print(domain, "déjà présent")
            continue
        path = hf_hub_download(
            repo_id="Salesforce/UniDoc-Bench",
            filename=f"{domain}_pdfs.tar.gz",
            repo_type="dataset",
        )
        with tarfile.open(path) as tar:
            tar.extractall(target)
        inner = target / domain
        if inner.is_dir():
            for f in os.listdir(inner):
                shutil.move(str(inner / f), str(target / f))
            inner.rmdir()
        for f in os.listdir(target):
            if f.startswith("._"):
                os.remove(target / f)
        for f in sorted(os.listdir(target))[MAX_PDFS_PER_DOMAIN:]:
            os.remove(target / f)
        print(domain, "->", len(os.listdir(target)), "PDF")


# ---------- 2. Chunking avec Spark ----------
def extract_text(pdf_path):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def process_partition(iterator):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    processed = 0
    for item in iterator:
        pdf_path, doc_id, domain = item["pdf_path"], item["doc_id"], item["domain"]
        try:
            text = extract_text(pdf_path)
            for k, chunk in enumerate(splitter.split_text(text)):
                yield (f"{doc_id}_c{k}", doc_id, domain, chunk)
            processed += 1
            print(f"[{domain}] {processed} PDF traités - dernier: {doc_id}", flush=True)
        except Exception as e:
            print(f"[ERREUR] {pdf_path}: {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def chunk_pdfs(spark):
    pdfs = []
    for p in RAW_DIR.rglob("*.pdf"):
        if p.name.startswith("._"):
            continue
        relative = p.relative_to(RAW_DIR)
        pdfs.append({
            "doc_id": "_".join(relative.with_suffix("").parts),
            "pdf_path": str(p.resolve()),
            "domain": relative.parts[0],
        })
    pdfs.sort(key=lambda x: x["pdf_path"])
    print(f"PDF trouvés : {len(pdfs)}")

    schema = StructType([
        StructField("chunk_id", StringType()),
        StructField("doc_id", StringType()),
        StructField("domain", StringType()),
        StructField("text", StringType()),
    ])

    pdf_rdd = spark.sparkContext.parallelize(pdfs, min(8, len(pdfs)))
    records_rdd = pdf_rdd.mapPartitions(process_partition).persist(StorageLevel.DISK_ONLY)
    df_chunks = spark.createDataFrame(records_rdd, schema)

    os.makedirs("data/chunks", exist_ok=True)
    df_chunks.write.mode("overwrite").partitionBy("domain").parquet(CHUNKS_DIR)
    records_rdd.unpersist()
    print("Chunking terminé !")


# ---------- 3. Embeddings avec Spark ----------
def embed_chunks(spark):
    @pandas_udf(ArrayType(FloatType()))
    def embed_udf(texts: pd.Series) -> pd.Series:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(EMBED_MODEL, device=device)
        emb = model.encode(texts.tolist(), batch_size=32)
        return pd.Series(emb.tolist())

    df_chunks = spark.read.parquet(CHUNKS_DIR).repartition(4)
    print(f"{df_chunks.count()} chunks à embedder")

    df_embedded = df_chunks.withColumn("embedding", embed_udf(df_chunks["text"]))
    os.makedirs("data/embeddings", exist_ok=True)
    df_embedded.write.mode("overwrite").partitionBy("domain").parquet(EMB_DIR)
    print("Embedding terminé !")


# ---------- 4. Index FAISS (VectorDB) ----------
def build_faiss():
    df = pd.read_parquet(EMB_DIR)
    print(f"{len(df)} chunks à indexer")

    embeddings = np.array(df["embedding"].tolist()).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    VDB_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VDB_DIR / "faiss.index"))
    df[["chunk_id", "doc_id", "domain", "text"]].to_pickle(VDB_DIR / "metadata.pkl")
    print(f"Index FAISS créé avec {index.ntotal} vecteurs")


def get_spark():
    spark = (
        SparkSession.builder
        .appName("RAG_Pipeline")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


if __name__ == "__main__":
    download_pdfs()
    spark = get_spark()
    chunk_pdfs(spark)
    embed_chunks(spark)
    spark.stop()
    build_faiss()