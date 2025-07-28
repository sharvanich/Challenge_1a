FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy your Python script into the container
COPY process_pdfs.py .

# Install dependencies
RUN pip install --no-cache-dir \
    PyMuPDF==1.23.5 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    scikit-image==0.21.0

# Set the default command
CMD ["python", "process_pdfs.py"]
