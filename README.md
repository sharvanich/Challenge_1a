# PDF Processing Solution - Adobe India Hackathon 2025 (Challenge 1a)

## 📝 Overview

This repository contains a solution for **Challenge 1a** of the **Adobe India Hackathon 2025**, which focuses on building a containerized PDF processing system. The system extracts structured data from input PDF files and outputs corresponding `.json` files adhering to a predefined schema.

---

## 🚀 Features

* ✅ Batch processing of PDFs from the `/app/input` directory
* ✅ Outputs structured JSON files to `/app/output`
* ✅ Containerized using Docker (AMD64 platform)
* ✅ No internet required during execution
* ✅ Compatible with CPU-only environments
* ✅ Optimized for speed and resource usage

---

## 📁 Directory Structure

```
Challenge_1a/
├── sample_dataset/
│   ├── pdfs/                 # Input PDF files (read-only)
│   ├── outputs/              # Output JSON files
│   └── schema/
│       └── output_schema.json  # Required output JSON schema
├── Dockerfile                # Docker container configuration
├── process_pdfs.py           # Main PDF processing script
└── README.md                 # This file
```

---

## ⚙️ System Requirements & Constraints

| Parameter       | Constraint                    |
| --------------- | ----------------------------- |
| Execution Time  | ≤ 10 seconds for 50-page PDF  |
| Max Model Size  | ≤ 200MB (if ML used)          |
| Runtime         | CPU-only (8 cores, 16 GB RAM) |
| Architecture    | AMD64 only                    |
| Internet Access | ❌ No access during execution  |

---

## 🐳 Docker Instructions

### Build

```bash
docker build --platform linux/amd64 -t pdf-processor .
```

### Run

```bash
docker run --rm \
-v $(pwd)/sample_dataset/pdfs:/app/input:ro \
-v $(pwd)/sample_dataset/outputs:/app/output \
--network none \
pdf-processor
```

---

## 📌 Expected Output

For every `filename.pdf` in the input folder, a structured file `filename.json` will be generated in the output folder.

✅ Output must conform to the schema:
`sample_dataset/schema/output_schema.json`

---

## 🧠 Implementation Details

The current `process_pdfs.py` is a placeholder demonstrating:

* Input file scanning
* Dummy data generation
* Output file writing

### What You Need to Improve

You should replace the dummy logic with:

* Actual PDF parsing (e.g., using `pdfminer.six`, `PyMuPDF`, or `pdfplumber`)
* Semantic segmentation into sections (title, headings, paragraphs, tables)
* Layout/structure parsing (multi-column, images)
* JSON generation conforming to schema

---

## ✅ Validation Checklist

* [x] All PDFs in `/app/input` are processed
* [x] Output is generated in `/app/output`
* [x] Output files follow the specified JSON schema
* [x] Processing finishes within time and memory limits
* [x] Code runs without any network access
* [x] Docker image compatible with AMD64 and CPU-only

---

## 🧪 Testing Strategy

* **Simple PDFs**: Plain text and headings
* **Complex PDFs**: Tables, images, columns
* **Large PDFs**: Ensure sub-10s processing for 50 pages

---

## 📚 Open Source Tools You May Use

* `pdfminer.six`, `pdfplumber`, `PyMuPDF` (for PDF parsing)
* `pydantic`, `jsonschema` (for schema validation)
* `multiprocessing`, `concurrent.futures` (for parallelism)

> 🔒 All tools and models used **must be open-source** and **offline-capable**.
