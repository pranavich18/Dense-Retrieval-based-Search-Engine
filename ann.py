import h5py
import numpy as np
import faiss
from tqdm import tqdm

# -----------------------------
# Load function (your format)
# -----------------------------
def load_h5_embeddings(file_path, id_key='id', embedding_key='embedding'):
    """
    Load IDs and embeddings from an HDF5 file.
    """
    print(f"Loading data from {file_path}...")
    with h5py.File(file_path, 'r') as f:
        ids = np.array(f[id_key]).astype(str)                # convert to string
        embeddings = np.array(f[embedding_key]).astype(np.float32)  # ensure float32
    print(f"Loaded {len(ids)} embeddings from {file_path}.")
    return ids, embeddings


# -----------------------------
# Load passage & query data
# -----------------------------
passage_ids, passage_embeddings = load_h5_embeddings(
    "ms_marco/msmarco_passages_embeddings_subset.h5",
)

query_ids, query_embeddings = load_h5_embeddings(
    "ms_marco/msmarco_passages_embeddings_subset.h5",
)


# -----------------------------
# Normalize for cosine similarity
# -----------------------------
faiss.normalize_L2(passage_embeddings)
faiss.normalize_L2(query_embeddings)


# -----------------------------
# Build FAISS HNSW Index
# -----------------------------
dim = passage_embeddings.shape[1]
M = 4  # graph connectivity
faiss.omp_set_num_threads(4)
index = faiss.IndexHNSWFlat(dim, M)

index.hnsw.efConstruction = 75  # indexing accuracy
index.hnsw.efSearch = 75        # query-time accuracy

print("Building HNSW index (this may take a few minutes)...")
index.add(passage_embeddings)


# -----------------------------
# Search Top-K documents
# -----------------------------
K = 1000   # retrieve enough for metrics like Recall@100 & NDCG@100
print("Running retrieval...")
scores, idx = index.search(query_embeddings, K)

retrieved_passage_ids = passage_ids[idx]  # map indices -> passage IDs


# -----------------------------
# Save TREC-formatted run file
# -----------------------------
run_file = "run_hnsw.txt"
print(f"Writing results to {run_file}...")

with open(run_file, "w") as f:
    for qi, qid in enumerate(query_ids):
        for rank, (pid, score) in enumerate(zip(retrieved_passage_ids[qi], scores[qi])):
            f.write(f"{qid} Q0 {pid} {rank+1} {score} hnsw_system\n")

print("Done. Run file ready for trec_eval.")
