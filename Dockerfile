FROM python:3.10-slim

WORKDIR /app

COPY process_pdfs.py .

RUN pip install --no-cache-dir \
    PyPDF2 \
    sentence-transformers \
    scikit-learn \
    networkx \
    numpy \
    matplotlib \
    leidenalg \
    igraph

CMD ["python", "process_pdfs.py"]