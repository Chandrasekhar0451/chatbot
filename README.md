# Doc Retrieval Flask Project (TF-IDF + ChromaDB)

Simple Flask-based document retrieval app with ingestion pipeline and semantic search.

## Project Structure

- `app.py`: Flask web app.
- `ingest.py`: Batch ingestion script.
- `utils.py`: File parsing/chunking helpers.
- `templates/`: HTML templates.
- `data/raw/knowledge/`: Source documents for ingestion.
- `requirements.txt`: Python dependencies.

## Quick Start

1. Create and activate virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Configure environment variables from `.env.example`.
4. Run app:
   `python app.py`

## Ingestion

Place documents in `data/raw/knowledge/`, then run:
`python ingest.py`
