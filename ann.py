import h5py
import numpy as np
import faiss
from tqdm import tqdm
import subprocess

# -----------------------------
# Load function
# -----------------------------
def load_h5_embeddings(file_path, id_key='id', embedding_key='embedding'):
    with h5py.File(file_path, 'r') as f:
        ids = np.array(f[id_key]).astype(str)
        embeddings = np.array(f[embedding_key]).astype(np.float32)
    return ids, embeddings

# -----------------------------
# Load passage & query data
# -----------------------------
passage_ids, passage_embeddings = load_h5_embeddings(
    "ms_marco/msmarco_passages_embeddings_subset.h5",
)
query_ids, query_embeddings = load_h5_embeddings(
    "ms_marco/msmarco_queries_dev_eval_embeddings.h5",
)


# Normalize embeddings for cosine similarity
faiss.normalize_L2(passage_embeddings)
faiss.normalize_L2(query_embeddings)

# Build FAISS HNSW index
dim = passage_embeddings.shape[1]
M = 32
index = faiss.IndexHNSWFlat(dim, M)
index.hnsw.efConstruction = 1000
index.hnsw.efSearch = 2000

faiss.omp_set_num_threads(faiss.omp_get_max_threads())

index.add(passage_embeddings)

K = 1000  # retrieve more docs per query
run_file = "run_hnsw.txt"
batch_size = 100  # adjust for RAM limits
num_batches = (len(query_embeddings) + batch_size - 1) // batch_size

with open(run_file, "w") as f:
    for start in tqdm(range(0, len(query_embeddings), batch_size), total=num_batches, desc="Retrieving batches"):
        batch = query_embeddings[start:start+batch_size]
        distances, labels = index.search(batch, K)

        for qi, qid in enumerate(query_ids[start:start+batch_size]):
            qid = str(qid).strip()
            for rank, (pid_idx, score) in enumerate(zip(labels[qi], distances[qi])):
                if rank >= 100:   # only keep top 100 per query
                    break
                pid = str(passage_ids[pid_idx]).strip()
                f.write(f"{qid} Q0 {pid} {rank+1} {score} hnsw_system\n")




