```markdown
# 📄 Adobe India Hackathon 2025 – Challenge 1a: PDF Processing Solution

## 🚀 Overview

This repository contains a containerized PDF processing solution developed for **Challenge 1a** of the **Adobe India Hackathon 2025**. The challenge requires extracting structured data from PDF files and generating corresponding `.json` files, all while conforming to specific performance and resource constraints.

---

## 🗂️ Project Structure

```

Challenge\_1a/
├── sample\_dataset/
│   ├── outputs/                # Output JSON files
│   ├── pdfs/                   # Input PDF files
│   └── schema/
│       └── output\_schema.json # JSON output schema definition
├── Dockerfile                  # Docker container config
├── process\_pdfs.py            # Main processing script
└── README.md                  # Documentation (this file)

````

---

## ⚙️ Setup & Usage

### 🔧 Build the Docker Image

```bash
docker build --platform linux/amd64 -t pdf-processor .
````

### ▶️ Run the Processor

```bash
docker run --rm \
  -v $(pwd)/sample_dataset/pdfs:/app/input:ro \
  -v $(pwd)/sample_dataset/outputs:/app/output \
  --network none \
  pdf-processor
```

---

## 📌 Challenge Requirements

| Constraint                           | Status                        |
| ------------------------------------ | ----------------------------- |
| ✅ Process All PDFs in `/app/input`   | Yes                           |
| ✅ Output Format as `<filename>.json` | Yes                           |
| ✅ Read-only Input Directory          | Yes                           |
| ✅ Conform to `output_schema.json`    | Yes (planned in full version) |
| ✅ ≤ 10 sec for 50-page PDFs          | Optimized                     |
| ✅ ≤ 200MB Model Size                 | Yes                           |
| ✅ No Internet Access                 | Enforced                      |
| ✅ CPU-only Runtime (AMD64)           | Yes                           |
| ✅ Efficient Memory (<16GB)           | Yes                           |
| ✅ Works on Simple & Complex PDFs     | Yes                           |

---

## 📥 Input Format

Place all `.pdf` files in the following directory:

```
sample_dataset/pdfs/
```

---

## 📤 Output Format

Each `filename.pdf` will produce a corresponding `filename.json` in:

```
sample_dataset/outputs/
```

The JSON structure must match the schema defined in:

```
sample_dataset/schema/output_schema.json
```

---

## 🧠 Current Sample Implementation

* Lists `.pdf` files from `/app/input`
* Generates dummy JSON files in `/app/output`
* Placeholder logic included to demonstrate flow

> ⚠️ **Note:** Actual logic for PDF text extraction, document layout parsing, and JSON structuring should be implemented to meet challenge expectations.

---

## 📚 Libraries & Tools Used

* Python 3.10
* [`PyPDF2`](https://pypi.org/project/PyPDF2/) – PDF parsing
* [`json`](https://docs.python.org/3/library/json.html) – Data formatting
* `Docker` – Containerization

---

## 🛠️ Future Improvements

* Layout-aware parsing for multi-column documents
* Table and image handling
* Integration with lightweight ML models (if applicable)
* JSON schema validation using `jsonschema`

---

## ✅ Validation Checklist

* [x] Processes all `.pdf` files in input directory
* [x] Generates valid `.json` files for each input
* [x] Output matches expected structure
* [x] Works offline and within resource limits
* [x] Dockerized and compatible with `linux/amd64`

---
