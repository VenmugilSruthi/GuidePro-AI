# rag.py (REPLACE with this)
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
from io import BytesIO

# optional OCR imports (only used if pure text extraction fails)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Try FAISS; if unavailable, fall back to numpy search
USE_FAISS = True
try:
    import faiss
except Exception:
    USE_FAISS = False

# Use a compact embedding model (env override allowed)
EMB_MODEL = os.getenv("RAG_MODEL", "all-MiniLM-L6-v2")

# LLM utils (your file must be in same package)
from llm_utils import get_llm_client, generate_answer

class RAGStore:
    def __init__(self, index_path="embeddings.pkl", docs_path="docs.pkl"):
        self.index_path = index_path
        self.docs_path = docs_path
        self.model = SentenceTransformer(EMB_MODEL)
        self.docs = []         # list of dicts: {"text":..., "source": filename}
        self.embs = None       # numpy (n, d) float32
        self.index = None      # faiss index if available

        # load existing
        if os.path.exists(self.docs_path) and os.path.exists(self.index_path):
            try:
                with open(self.docs_path, "rb") as f:
                    self.docs = pickle.load(f)
                with open(self.index_path, "rb") as f:
                    self.embs = pickle.load(f)
            except Exception:
                self.docs = []
                self.embs = None

        # if faiss available and embeddings present, build IP index
        if USE_FAISS and self.embs is not None:
            d = self.embs.shape[1]
            # ensure float32
            self.embs = self.embs.astype("float32")
            self.index = faiss.IndexFlatIP(d)
            # embeddings must be normalized for cosine similarity via inner product
            faiss.normalize_L2(self.embs)
            self.index.add(self.embs)

    def _extract_text(self, filelike):
        # try PyPDF2 extraction
        try:
            filelike.seek(0)
            reader = PdfReader(filelike)
            texts = []
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    texts.append(t)
            combined = "\n".join(texts).strip()
            if combined:
                return combined
        except Exception:
            pass

        # fallback to OCR if available
        if OCR_AVAILABLE:
            try:
                filelike.seek(0)
                imgs = convert_from_bytes(filelike.read())
                text = ""
                for im in imgs:
                    text += pytesseract.image_to_string(im)
                return text
            except Exception:
                pass

        return ""

    def _chunk_text(self, text, filename="<doc>", chunk_size=400, chunk_overlap=100):
        # chunk by words with overlap to keep context coherent
        words = text.split()
        if not words:
            return []
        chunks = []
        step = chunk_size - chunk_overlap
        i = 0
        while i < len(words):
            chunk_words = words[i:i+chunk_size]
            chunk = " ".join(chunk_words).strip()
            if len(chunk) > 30:  # filter tiny chunks
                chunks.append({"text": chunk, "source": filename})
            i += max(1, step)
        return chunks

    def _save(self):
        with open(self.docs_path, "wb") as f:
            pickle.dump(self.docs, f)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.embs, f)

    def add_documents(self, uploaded_files):
        """
        uploaded_files: iterable of file-like objects (with .name optionally)
        """
        added_any = False
        new_texts = []
        for f in uploaded_files:
            try:
                filename = getattr(f, "name", "<uploaded>")
                f.seek(0)
                text = self._extract_text(f)
                if not text or not text.strip():
                    continue
                chunks = self._chunk_text(text, filename=filename)
                if not chunks:
                    continue
                texts = [c["text"] for c in chunks]

                # embed and normalize
                embs = self.model.encode(texts, convert_to_numpy=True)
                embs = embs.astype("float32")
                faiss.normalize_L2(embs)

                # append to existing
                if self.embs is None:
                    self.embs = embs
                else:
                    self.embs = np.vstack([self.embs, embs])

                self.docs.extend(chunks)
                added_any = True
                new_texts.extend(texts)

            except Exception as e:
                print("Error adding document:", e)

        if added_any:
            # rebuild faiss index for correctness
            if USE_FAISS and self.embs is not None:
                d = self.embs.shape[1]
                self.index = faiss.IndexFlatIP(d)
                # faiss expects normalized vectors for inner product = cosine
                faiss.normalize_L2(self.embs)
                self.index.add(self.embs)
            # save to disk
            self._save()

    def _numpy_search(self, q_emb, top_k):
        # cosine with pre-normalized self.embs (if available)
        if self.embs is None:
            return []
        # ensure shapes
        emb_matrix = self.embs
        # if not normalized, normalize temporarily
        if np.max(np.linalg.norm(emb_matrix, axis=1)) > 1.0001:
            emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9)
        qvec = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        sims = emb_matrix @ qvec
        idxs = list(np.argsort(-sims)[:top_k])
        return idxs

    def query(self, q, top_k=3, answer_with_llm=True):
        """
        q: user question string
        top_k: number of chunks to retrieve
        answer_with_llm: if True, call LLM to synthesize a short answer using retrieved context
        """
        if not self.docs or self.embs is None:
            return "No documents indexed yet. Upload PDFs."

        # get query embedding
        q_emb = self.model.encode([q], convert_to_numpy=True)[0].astype("float32")
        # normalize q_emb for cosine+IP
        q_emb_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)

        # use faiss if available
        if USE_FAISS and self.index is not None:
            # faiss expects a 2D array and normalized embeddings
            q_search = np.expand_dims(q_emb_norm, axis=0).astype("float32")
            D, I = self.index.search(q_search, top_k)
            idxs = [int(i) for i in I[0] if i >= -1]
        else:
            idxs = self._numpy_search(q_emb_norm, top_k)

        # collect results and dedupe near-duplicates
        results = []
        seen_texts = set()
        for idx in idxs:
            if 0 <= idx < len(self.docs):
                txt = self.docs[idx]["text"]
                if txt not in seen_texts:
                    results.append(self.docs[idx])
                    seen_texts.add(txt)

        if not results:
            return "No relevant information found."

        # Build a short context by concatenating top results (but cap total length)
        max_chars = 3000
        pieces = []
        for r in results:
            pieces.append(f"[source: {r.get('source','-')}] {r['text']}")
            if sum(len(p) for p in pieces) > max_chars:
                break
        context_for_llm = "\n\n---\n\n".join(pieces)

        if not answer_with_llm:
            # Just return raw chunks (for debugging) but trimmed
            return "\n\n".join([r["text"] for r in results])

        # Use your llm_utils to synthesize final answer
        client = get_llm_client()
        if client is None:
            # fallback: return concatenated context if LLM not configured
            return context_for_llm

        # craft a concise prompt
        system_msg = {
            "role": "system",
            "content": (
                "You are GuidePro AI. Given the provided document context, answer the user's question concisely. "
                "Do NOT repeat the entire documents. If the answer is not present, say 'Not found in uploaded documents.' "
                "Cite sources in square brackets like [source: filename]. Keep answer under 200 words."
            )
        }
        user_msg = {
            "role": "user",
            "content": f"QUESTION: {q}\n\nCONTEXT:\n{context_for_llm}\n\nProvide a concise answer and cite sources if used."
        }
        messages = [system_msg, user_msg]

        try:
            answer = generate_answer(client, messages)
            # safety: if LLM simply echoes the context, shorten
            if len(answer) > 2000 and "No relevant information found" not in answer:
                # return short summary using sentence-transformers summarization fallback
                answer = answer[:1500] + "..."
            return answer
        except Exception as e:
            print("LLM call failed:", e)
            return context_for_llm
