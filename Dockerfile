FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
