# Troubleshooting Guide

## Ollama not connecting

**Symptom:** Health check shows `ollama_connected: false`.

**Fix:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# In Docker: check the ollama container
docker compose logs ollama
```

---

## Models not pulled

**Symptom:** Ingestion or chat fails with a model error.

**Fix:**
```bash
ollama pull llama3
ollama pull nomic-embed-text

# Verify
ollama list
```

---

## Ingestion fails with PDF errors

**Symptom:** Document status is `error`, message mentions PDF parsing.

**Fix:** Some PDFs are scanned images (no text layer). Use a tool like `ocrmypdf` first:
```bash
pip install ocrmypdf
ocrmypdf input.pdf output.pdf
```

---

## ChromaDB permission error

**Symptom:** `PermissionError` on `./data/chroma`.

**Fix:**
```bash
chmod -R 777 ./data/chroma
# Or in Docker:
docker compose down
sudo chown -R $USER:$USER ./data
docker compose up -d
```

---

## Frontend shows "Ollama offline" banner

This means the backend can't reach Ollama. In Docker Compose, make sure Ollama is healthy:
```bash
docker compose ps
docker compose logs ollama --tail=50
```

---

## Very slow responses

- Use a smaller model: switch `chat_model` to `phi3` or `gemma2:2b`
- Reduce `retrieval.top_k` from 5 to 3
- Ensure Ollama is using GPU if available (see docker-compose.yml GPU section)

---

## Database locked error (SQLite)

SQLite allows only one writer. The app uses `workers: 1` in uvicorn to avoid this. If you see this error, verify:
```yaml
# config/settings.yaml
server:
  workers: 1  # Must be 1 for SQLite
```

---

## Port already in use

```bash
# Find what's using port 8000
lsof -i :8000
# Kill it or change the port in docker-compose.yml
```

---

## Reset everything

```bash
docker compose down -v          # Remove containers + volumes
rm -rf ./data/chroma ./data/db  # Delete local data
docker compose up -d            # Fresh start
```
