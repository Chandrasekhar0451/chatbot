from flask import Flask, jsonify, render_template, request
import chromadb
from chromadb.utils import embedding_functions
import os
import re
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from utils import yield_file_chunks

load_dotenv()
openai_client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
)



app = Flask(__name__)
# Keep upload size configurable via env var, defaulting to 300 MB.
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "300")) * 1024 * 1024
UPLOAD_INGEST_BATCH_SIZE = int(os.getenv("UPLOAD_INGEST_BATCH_SIZE", "100"))
ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".pdf", ".docx"}
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "data/raw/knowledge")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)


def get_embedding_function():
    print("[get_embedding_function] Function called")
    backend = os.getenv("EMBEDDING_BACKEND", "default").strip().lower()
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
    print(f"[get_embedding_function] backend={backend}, model_name={model_name}")

    if backend in {"default", "onnx"}:
        print("[get_embedding_function] Using DefaultEmbeddingFunction (ONNX)")
        return embedding_functions.DefaultEmbeddingFunction()

    try:
        print(f"[get_embedding_function] Loading SentenceTransformer model: {model_name}")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        print("[get_embedding_function] SentenceTransformer loaded successfully")
        return ef
    except BaseException as e:
        print(f"Warning: failed to initialize sentence-transformer embedding ({e}). Falling back to default embedding.")
        return embedding_functions.DefaultEmbeddingFunction()

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with", "you", "your"
}

print("[STARTUP] Initializing ChromaDB in app...")
chroma_client = chromadb.PersistentClient(path="index/chroma_db")
print("[STARTUP] ChromaDB PersistentClient created successfully")
embedding_ef = get_embedding_function()
print("[STARTUP] Embedding function ready")

try:
    collection = chroma_client.get_collection(
        name="knowledge_base",
        embedding_function=embedding_ef
    )
    print("[STARTUP] Existing collection 'knowledge_base' loaded successfully")
except Exception as e:
    print(f"[STARTUP] Warning: ChromaDB collection not found yet. It will be created on first upload. Error: {e}")
    collection = None

def ensure_collection():
    global collection
    print("[ensure_collection] Function called")
    if collection is None:
        print("[ensure_collection] Collection is None, creating new collection...")
        collection = chroma_client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=embedding_ef
        )
        print("[ensure_collection] Collection created successfully")
    else:
        print("[ensure_collection] Collection already exists")
    return collection

def list_knowledge_files():
    files = []
    for name in os.listdir(KNOWLEDGE_DIR):
        path = os.path.join(KNOWLEDGE_DIR, name)
        if os.path.isfile(path):
            files.append(name)
    files.sort(key=lambda x: x.lower())
    return files

def ingest_uploaded_file(uploaded_file):
    print("[ingest_uploaded_file] ========== FUNCTION CALLED ==========")
    if uploaded_file is None:
        print("[ingest_uploaded_file] ERROR: uploaded_file is None")
        return False, "Please select a file to upload."

    print(f"[ingest_uploaded_file] Received file: {uploaded_file.filename}")
    filename = secure_filename(uploaded_file.filename or "")
    if not filename:
        print("[ingest_uploaded_file] ERROR: filename is empty after secure_filename")
        return False, "Please select a file to upload."

    ext = os.path.splitext(filename)[1].lower()
    print(f"[ingest_uploaded_file] Filename: {filename}, Extension: {ext}")
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        print(f"[ingest_uploaded_file] ERROR: Extension '{ext}' not allowed")
        return False, f"Unsupported file type. Allowed types: {allowed}"

    saved_path = os.path.join(KNOWLEDGE_DIR, filename)
    print(f"[ingest_uploaded_file] Saving file to: {saved_path}")
    try:
        uploaded_file.stream.seek(0)
        uploaded_file.save(saved_path)
        print(f"[ingest_uploaded_file] File saved successfully. Size: {os.path.getsize(saved_path)} bytes")
    except Exception as e:
        print(f"[ingest_uploaded_file] ERROR saving file: {e}")
        return False, f"Failed to save file: {e}"

    print("[ingest_uploaded_file] Getting/creating collection...")
    target_collection = ensure_collection()
    print("[ingest_uploaded_file] Collection ready")

    print(f"[ingest_uploaded_file] Checking for existing chunks with source='{filename}'...")
    existing = target_collection.get(where={"source": filename}, include=[])
    existing_ids = existing.get("ids", []) if existing else []
    if existing_ids:
        print(f"[ingest_uploaded_file] Deleting {len(existing_ids)} existing chunks for this file")
        target_collection.delete(ids=existing_ids)
        print("[ingest_uploaded_file] Old chunks deleted")
    else:
        print("[ingest_uploaded_file] No existing chunks found for this file")

    batch_documents = []
    batch_metadatas = []
    batch_ids = []
    chunk_count = 0

    print(f"[ingest_uploaded_file] Starting chunking process for: {saved_path}")
    try:
        for chunk in yield_file_chunks(saved_path):
            chunk = (chunk or "").strip()
            if not chunk:
                continue

            batch_documents.append(chunk)
            batch_metadatas.append({"source": filename})
            batch_ids.append(f"{filename}_{uuid.uuid4().hex}_{chunk_count}")
            chunk_count += 1

            if len(batch_documents) >= UPLOAD_INGEST_BATCH_SIZE:
                print(f"[ingest_uploaded_file] Upserting batch of {len(batch_documents)} chunks (total so far: {chunk_count})...")
                try:
                    target_collection.upsert(
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    print(f"[ingest_uploaded_file] Batch upsert successful")
                except Exception as e:
                    print(f"[ingest_uploaded_file] ERROR during batch upsert: {e}")
                    return False, f"Failed during embedding/indexing: {e}"
                batch_documents = []
                batch_metadatas = []
                batch_ids = []
    except Exception as e:
        print(f"[ingest_uploaded_file] ERROR during chunking: {e}")
        return False, f"Failed during file chunking: {e}"

    print(f"[ingest_uploaded_file] Chunking complete. Total chunks: {chunk_count}")

    if batch_documents:
        print(f"[ingest_uploaded_file] Upserting final batch of {len(batch_documents)} chunks...")
        try:
            target_collection.upsert(
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print("[ingest_uploaded_file] Final batch upsert successful")
        except Exception as e:
            print(f"[ingest_uploaded_file] ERROR during final batch upsert: {e}")
            return False, f"Failed during final embedding/indexing: {e}"

    if chunk_count == 0:
        print("[ingest_uploaded_file] WARNING: No readable text found in the file")
        return False, "No readable text found in the uploaded file."

    print(f"[ingest_uploaded_file] ========== SUCCESS: {chunk_count} chunks indexed ==========")
    return True, f"Uploaded and indexed '{filename}' with {chunk_count} chunks."

@app.route("/upload", methods=["POST"])
def upload():
    print("\n[upload] >>>>>>>>>> /upload endpoint hit <<<<<<<<<<")
    print(f"[upload] Request content length: {request.content_length}")
    print(f"[upload] Request content type: {request.content_type}")
    try:
        uploaded_file = request.files.get("upload_file")
        print(f"[upload] File received: {uploaded_file.filename if uploaded_file else 'None'}")
    except Exception as e:
        print(f"[upload] ERROR getting file from request: {e}")
        return jsonify({"ok": False, "message": f"Error reading upload: {e}", "uploaded_files": list_knowledge_files()}), 400

    ok, message = ingest_uploaded_file(uploaded_file)
    status_code = 200 if ok else 400
    print(f"[upload] Result: ok={ok}, message={message}")
    print("[upload] >>>>>>>>>> /upload endpoint finished <<<<<<<<<<\n")
    return jsonify(
        {
            "ok": ok,
            "message": message,
            "uploaded_files": list_knowledge_files(),
        }
    ), status_code

@app.route("/knowledge-files", methods=["GET"])
def knowledge_files():
    return jsonify({"uploaded_files": list_knowledge_files()}), 200

def generate_rag_answer(chunks, query):
    print(f"[generate_rag_answer] Function called with {len(chunks)} chunks")
    if not chunks:
        print("[generate_rag_answer] No chunks provided, returning default message")
        return "Sorry, I couldn't find any relevant documents to answer your question."

    # Combine all retrieved text to serve as context
    context = "\n\n".join([chunk["text"].strip() for chunk in chunks])
    print(f"[generate_rag_answer] Context length: {len(context)} characters")

    prompt = f"""You are a helpful, intelligent Document Retrieval Assistant. 
Please read the provided Excerpts and answer the User's Question clearly and conversationally.
- Use ONLY the provided Excerpts.
- If the answer is not contained in the Excerpts, simply reply: "I'm sorry, I don't see the answer to that in the provided documents."
- If the Excerpts are not relevant to the question, do not guess.

Excerpts:
{context}

User's Question: {query}
Answer:"""

    try:
        print(f"[generate_rag_answer] Calling LLM API (model: {os.getenv('XAI_MODEL', 'grok-3-latest')})...")
        response = openai_client.chat.completions.create(
            model=os.getenv("XAI_MODEL", "grok-3-latest"),
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful document retrieval assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        print("[generate_rag_answer] LLM API response received successfully")
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"[generate_rag_answer] ERROR - LLM Generation Error: {e}")
        # Fallback to the raw chunks if the API call fails
        seen = set()
        paragraphs = []
        for chunk in chunks:
            text = chunk["text"].strip()
            if text and text not in seen:
                seen.add(text)
                paragraphs.append(text)
        return "\n\n".join(paragraphs)

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def rerank_and_filter_chunks(raw_chunks, query, top_k, max_distance):
    query_terms = {tok for tok in tokenize(query) if tok not in STOPWORDS}
    ranked = []

    for chunk in raw_chunks:
        dist = chunk["distance"]
        if dist > max_distance:
            continue

        chunk_terms = set(tokenize(chunk["text"]))
        overlap = 0.0
        if query_terms:
            overlap = len(query_terms & chunk_terms) / len(query_terms)

        semantic_score = max(0.0, 1.0 - (dist / max_distance))
        hybrid_score = 0.8 * semantic_score + 0.2 * overlap

        # Reject only very weak matches; keep moderately related context.
        if overlap == 0 and dist > (max_distance * 0.9):
            continue

        ranked.append(
            {
                "source": chunk["source"],
                "score": round(dist, 3),
                "text": chunk["text"],
                "hybrid_score": hybrid_score,
            }
        )

    ranked.sort(key=lambda x: (-x["hybrid_score"], x["score"]))

    deduped = []
    seen_texts = set()
    for item in ranked:
        key = " ".join(item["text"].split()).lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        item.pop("hybrid_score", None)
        deduped.append(item)
        if len(deduped) >= top_k:
            break

    if deduped:
        return deduped

    # Fallback: if filters were too strict, return nearest chunks by distance.
    raw_by_distance = sorted(raw_chunks, key=lambda x: x["distance"])
    fallback = []
    for item in raw_by_distance[:top_k]:
        fallback.append(
            {
                "source": item["source"],
                "score": round(item["distance"], 3),
                "text": item["text"],
            }
        )
    return fallback

def search(query, top_k=None):
    print(f"[search] Function called with query: '{query[:50]}...'")
    if collection is None:
        print("[search] ERROR: Collection is None, search unavailable")
        return {"answer": "Search is currently unavailable because the index has not been built."}

    if top_k is None:
        top_k = int(os.getenv("RETRIEVAL_TOP_K", "8"))

    initial_k = max(top_k * 4, 20)
    max_distance = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "1.4"))
    print(f"[search] top_k={top_k}, initial_k={initial_k}, max_distance={max_distance}")

    print("[search] Querying ChromaDB...")
    results = collection.query(
        query_texts=[query],
        n_results=initial_k
    )
    print(f"[search] ChromaDB returned {len(results['documents'][0]) if results and results['documents'] and results['documents'][0] else 0} results")

    raw_chunks = []
    if results and results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            doc_text = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            # ChromaDB returns distance: lower is better
            dist = float(results['distances'][0][i])

            raw_chunks.append({
                "source": meta.get("source", "Unknown"),
                "distance": dist,
                "text": doc_text,
            })

    print(f"[search] Reranking {len(raw_chunks)} chunks...")
    chunks = rerank_and_filter_chunks(raw_chunks, query, top_k=top_k, max_distance=max_distance)
    print(f"[search] After reranking: {len(chunks)} chunks")

    print("[search] Generating RAG answer...")
    answer = generate_rag_answer(chunks, query)
    print("[search] Answer generated successfully")

    return {
        "answer": answer,
        "chunks": chunks
    }

@app.route("/", methods=["GET", "POST"])
def index():
    print(f"[index] / endpoint hit, method={request.method}")
    query = ""
    answer = ""
    chunks = []
    upload_message = ""
    upload_error = ""

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        print(f"[index] POST query: '{query[:50]}...' " if query else "[index] POST with empty query")
        if query:
            result = search(query)
            answer = result.get("answer", "")
            chunks = result.get("chunks", [])
            print(f"[index] Search returned {len(chunks)} chunks")

    print("[index] Rendering template")
    return render_template(
        "index.html",
        query=query,
        answer=answer,
        chunks=chunks,
        uploaded_files=list_knowledge_files(),
        upload_message=upload_message,
        upload_error=upload_error
    )

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    limit_mb = int(app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024))
    message = f"File is too large. Maximum allowed size is {limit_mb} MB."

    if request.path == "/upload":
        return jsonify(
            {
                "ok": False,
                "message": message,
                "uploaded_files": list_knowledge_files(),
            }
        ), 413

    return render_template(
        "index.html",
        query="",
        answer="",
        chunks=[],
        uploaded_files=list_knowledge_files(),
        upload_message="",
        upload_error=message
    ), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
