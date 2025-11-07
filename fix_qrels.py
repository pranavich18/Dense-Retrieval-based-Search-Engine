# fix_qrels.py
input_file = "qrels.dev.tsv"
output_file = "qrels.dev.trec.tsv"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        line = line.strip()
        if not line or line.lower().startswith("query"):  # skip header
            continue
        parts = line.split()
        if len(parts) != 3:
            continue  # skip malformed lines
        query_id, doc_id, relevance = parts
        f_out.write(f"{query_id} 0 {doc_id} {relevance}\n")

print(f"Fixed qrels saved to {output_file}")