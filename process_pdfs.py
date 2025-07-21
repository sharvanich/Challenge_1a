from pathlib import Path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import igraph as ig
import leidenalg
import json
import os

# Base path (change accordingly if needed)
base_path = Path("C:/Users/Srinu/Downloads/sample_dataset")

input_dir = base_path / "pdfs"
output_dir = base_path / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

def classify_line(text, cluster_id, cluster_size):
    if cluster_size < 3 and len(text.split()) < 6:
        return "Title"
    elif cluster_size < 6 and len(text.split()) < 10:
        return "Heading"
    elif len(text.split()) < 15:
        return "Subheading"
    else:
        return "Paragraph"

def process_pdf(pdf_path, output_path):
    reader = PdfReader(pdf_path)
    lines = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            lines.extend(text.strip().split("\n"))

    if not lines:
        with open(output_path, "w") as f:
            json.dump({"content": []}, f)
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(lines)
    sim_matrix = cosine_similarity(embeddings)

    G_nx = nx.Graph()
    for i, line in enumerate(lines):
        G_nx.add_node(i, text=line)

    threshold = 0.7
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if sim_matrix[i][j] > threshold:
                G_nx.add_edge(i, j, weight=sim_matrix[i][j])

    edges = [(u, v, d["weight"]) for u, v, d in G_nx.edges(data=True)]
    if not edges:
        output_content = [{"text": line, "label": "Paragraph", "cluster_id": -1} for line in lines]
    else:
        G_ig = ig.Graph.TupleList(edges, edge_attrs=["weight"])
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)

        output_content = []
        for cluster_id, cluster in enumerate(partition):
            for i in cluster:
                line_text = lines[i]
                label = classify_line(line_text, cluster_id, len(cluster))
                output_content.append({
                    "text": line_text,
                    "label": label,
                    "cluster_id": cluster_id
                })

    with open(output_path, "w") as f:
        json.dump({"content": output_content}, f, indent=2)

def process_all_pdfs():
    for pdf_file in input_dir.glob("*"):
        if pdf_file.suffix.lower() != ".pdf":
            continue
        print(f"Processing {pdf_file.name}...")
        output_file = output_dir / f"{pdf_file.stem}.json"
        process_pdf(pdf_file, output_file)

if __name__ == "__main__":
    process_all_pdfs()