# ------ Build React App ------
FROM node:20 AS frontend-builder

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./

RUN npm install

COPY frontend/ ./

RUN npm run build

# ------ Build FastAPI App ------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git gcc g++ libffi-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY .env .env

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
# ------------------------------------------------------
COPY --from=frontend-builder /app/dist ./frontend/dist

EXPOSE 8080

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "backend.main:app", "-b", "0.0.0.0:8080"]
