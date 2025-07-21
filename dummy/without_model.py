import fitz  # PyMuPDF
import json
from collections import defaultdict

# Load PDF
doc = fitz.open("Travigenie.pdf")

# Step 1: Extract all text blocks with font size and page info
all_lines = []

for page_num, page in enumerate(doc, start=1):
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                line_text = ""
                font_sizes = []
                bold_flags = []
                for span in line["spans"]:
                    line_text += span["text"]
                    font_sizes.append(span["size"])
                    bold_flags.append("Bold" in span["font"])
                if line_text.strip():
                    avg_font = sum(font_sizes) / len(font_sizes)
                    is_bold = any(bold_flags)
                    all_lines.append({
                        "text": line_text.strip(),
                        "page": page_num,
                        "font_size": round(avg_font, 2),
                        "bold": is_bold
                    })

# Step 2: Group by font size to identify heading levels
font_size_counts = defaultdict(int)
for line in all_lines:
    font_size_counts[line["font_size"]] += 1

# Sort font sizes descending and assign levels
sorted_sizes = sorted(font_size_counts.items(), key=lambda x: -x[0])
font_to_level = {}
for i, (size, _) in enumerate(sorted_sizes):
    if i == 0:
        font_to_level[size] = "H1"
    elif i == 1:
        font_to_level[size] = "H2"
    elif i == 2:
        font_to_level[size] = "H3"
    else:
        font_to_level[size] = "Paragraph"

# Step 3: Build structured output
output = {
    "title": None,
    "outline": []
}

for line in all_lines:
    level = font_to_level[line["font_size"]]
    entry = {
        "level": level,
        "text": line["text"],
        "page": line["page"]
    }
    if level == "H1" and not output["title"]:
        output["title"] = line["text"]
    output["outline"].append(entry)

# Step 4: Print output
print("\n📄 Document Structure:\n")
for item in output["outline"]:
    print(f"[{item['level']}] (Page {item['page']}): {item['text']}")

# # Step 5: Save to JSON
# with open("output_structure.json", "w", encoding="utf-8") as f:
#     json.dump(output, f, indent=4, ensure_ascii=False)

# print("\n✅ Structured output saved to output_structure.json")
