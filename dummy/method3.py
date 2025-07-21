import json
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import igraph as ig
import leidenalg
import re

# Step 1: Extract lines from PDF
doc = fitz.open("Travigenie.pdf")
lines = []
pages = []

for page_num, page in enumerate(doc, start=1):
    text = page.get_text()
    if text:
        for line in text.strip().split("\n"):
            clean_line = line.strip()
            if len(clean_line.split()) >= 2 and not re.match(r"^(https?:\/\/|www\.|%?pip install)", clean_line.lower()):
                lines.append(clean_line)
                pages.append(page_num)

doc.close()

if not lines:
    print("❌ No valid lines extracted.")
    exit()
else:
    print(f"✅ Extracted {len(lines)} valid lines.")

# Step 2: Embed lines
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(lines)

if not embeddings.size > 0:
    print("❌ No embeddings generated.")
    exit()

# Step 3: Graph + similarity
sim_matrix = cosine_similarity(embeddings)
threshold = 0.6
G_nx = nx.Graph()
for i in range(len(lines)):
    G_nx.add_node(i)
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        if sim_matrix[i][j] > threshold:
            G_nx.add_edge(i, j, weight=sim_matrix[i][j])

# Step 4: Leiden Clustering
edges = [(u, v, d["weight"]) for u, v, d in G_nx.edges(data=True)]
G_ig = ig.Graph.TupleList(edges, edge_attrs=["weight"])
partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)

# Step 5: Rank clusters
cluster_sizes = sorted([(i, len(c)) for i, c in enumerate(partition)], key=lambda x: -x[1])
cluster_rank = {cluster_idx: rank for rank, (cluster_idx, _) in enumerate(cluster_sizes)}

# Step 6: Classification function
def classify_line(line, rank):
    line = line.strip()
    if rank == 0 and len(line.split()) <= 12:
        return "Title"
    elif line.endswith(":") or (line.istitle() and len(line.split()) <= 8):
        return "Heading"
    elif len(line.split()) <= 5:
        return "Subheading"
    else:
        return "Paragraph"

# Step 7: Annotate every line with its cluster + role
line_annotations = []
for cluster_idx, cluster in enumerate(partition):
    rank = cluster_rank[cluster_idx]
    for idx in cluster:
        level = classify_line(lines[idx], rank)
        line_annotations.append({
            "level": level,
            "text": lines[idx],
            "page": pages[idx]
        })

# Step 8: Print structured hierarchy
print("\n📄 Structured Hierarchy:\n")
for entry in line_annotations:
    print(f"[{entry['level']}] (Page {entry['page']}): {entry['text']}")

# Optional: Save to JSON
# with open("full_output_structure.json", "w", encoding="utf-8") as f:
#     json.dump(line_annotations, f, indent=4, ensure_ascii=False)