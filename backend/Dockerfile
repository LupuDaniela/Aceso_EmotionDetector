FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN grep -v "^torch" requirements.txt > requirements_notorch.txt && \
    pip install --no-cache-dir -r requirements_notorch.txt

RUN pip install gdown

COPY . .

RUN gdown "1YmDOD28_TD62yXHQUX1cEGlmEMxDlZTc" -O models/best_model_3.pt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]