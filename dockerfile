FROM python:3.9-slim as builder

WORKDIR /install

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt

FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ /app/src/
COPY setup.py .

RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "production_lstm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]