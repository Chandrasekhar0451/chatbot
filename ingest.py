import os
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from utils import yield_file_chunks

DATA_DIR = "data/raw/knowledge"
INDEX_DIR = "index/chroma_db"

os.makedirs(INDEX_DIR, exist_ok=True)


def get_embedding_function():
    backend = os.getenv("EMBEDDING_BACKEND", "default").strip().lower()
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()

    if backend in {"default", "onnx"}:
        return embedding_functions.DefaultEmbeddingFunction()

    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    except BaseException as e:
        print(f"Warning: failed to initialize sentence-transformer embedding ({e}). Falling back to default embedding.")
        return embedding_functions.DefaultEmbeddingFunction()

# Initialize ChromaDB client
print("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path=INDEX_DIR)
embedding_ef = get_embedding_function()

collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=embedding_ef
)

# Get existing documents to avoid re-ingesting
existing_docs = collection.get(include=["metadatas"])
existing_sources = set()
if existing_docs and existing_docs["metadatas"]:
    for meta in existing_docs["metadatas"]:
        if meta and "source" in meta:
             existing_sources.add(meta["source"])

new_files_added = False
BATCH_SIZE = 100

for filename in tqdm(os.listdir(DATA_DIR)):
    if filename in existing_sources:
        continue # Skip already processed files
        
    path = os.path.join(DATA_DIR, filename)
    
    # Process the file in a streaming way
    chunk_generator = yield_file_chunks(path)
    if chunk_generator is None:
        continue
        
    batch_documents = []
    batch_metadatas = []
    batch_ids = []
    
    chunk_idx = 0
    for chunk in chunk_generator:
        if not chunk.strip():
            continue
            
        batch_documents.append(chunk)
        batch_metadatas.append({"source": filename})
        # Create a unique ID for each chunk
        batch_ids.append(f"{filename}_chunk_{chunk_idx}")
        chunk_idx += 1
        
        # Upsert in batches to keep memory usage low
        if len(batch_documents) >= BATCH_SIZE:
            collection.upsert(
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            batch_documents = []
            batch_metadatas = []
            batch_ids = []
            new_files_added = True

    # Upsert any remaining chunks for this file
    if batch_documents:
        collection.upsert(
            documents=batch_documents,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        new_files_added = True

if not new_files_added:
    print("No new files to ingest. Index is up to date.")
else:
    print("Index updated successfully.")
    print("Total chunks in database:", collection.count())
