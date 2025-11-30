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

from llm_utils import get_llm_client, generate_answer

class RAGStore:
    def __init__(self, index_path="embeddings.pkl", docs_path="docs.pkl"):
        self.index_path = index_path
        self.docs_path = docs_path
        self.model = SentenceTransformer(EMB_MODEL)
        self.docs = []
        self.embs = None
        self.index = None

        # Load existing
        if os.path.exists(self.docs_path) and os.path.exists(self.index_path):
            try:
                with open(self.docs_path, "rb") as f:
                    self.docs = pickle.load(f)
                with open(self.index_path, "rb") as f:
                    self.embs = pickle.load(f)
            except Exception:
                self.docs = []
                self.embs = None

        # Build FAISS index
        if USE_FAISS and self.embs is not None:
            d = self.embs.shape[1]
            self.embs = self.embs.astype("float32")
            self.index = faiss.IndexFlatIP(d)
            faiss.normalize_L2(self.embs)
            self.index.add(self.embs)

    def _extract_text(self, filelike):
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

        # OCR fallback
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
        words = text.split()
        if not words:
            return []
        chunks = []
        step = chunk_size - chunk_overlap

        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words).strip()
            if len(chunk) > 30:
                chunks.append({"text": chunk, "source": filename})
            i += max(1, step)

        return chunks

    def _save(self):
        with open(self.docs_path, "wb") as f:
            pickle.dump(self.docs, f)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.embs, f)

    def add_documents(self, uploaded_files):
        added_any = False

        for f in uploaded_files:
            try:
                filename = getattr(f, "name", "<uploaded>")
                f.seek(0)
                text = self._extract_text(f)

                if not text.strip():
                    continue

                chunks = self._chunk_text(text, filename)
                if not chunks:
                    continue

                texts = [c["text"] for c in chunks]
                embs = self.model.encode(texts, convert_to_numpy=True).astype("float32")
                faiss.normalize_L2(embs)

                if self.embs is None:
                    self.embs = embs
                else:
                    self.embs = np.vstack([self.embs, embs])

                self.docs.extend(chunks)
                added_any = True

            except Exception as e:
                print("Error adding document:", e)

        if added_any:
            if USE_FAISS and self.embs is not None:
                d = self.embs.shape[1]
                self.index = faiss.IndexFlatIP(d)
                faiss.normalize_L2(self.embs)
                self.index.add(self.embs)

            self._save()

    def _numpy_search(self, q_emb, top_k):
        if self.embs is None:
            return []

        emb_matrix = self.embs
        if np.max(np.linalg.norm(emb_matrix, axis=1)) > 1.0001:
            emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9)

        qvec = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        sims = emb_matrix @ qvec
        idxs = np.argsort(-sims)[:top_k]
        return idxs.tolist()

    def query(self, q, top_k=3, answer_with_llm=True):

        # FIX: Disable RAG if no PDFs uploaded
        if not self.docs or self.embs is None:
            return None

        q_emb = self.model.encode([q], convert_to_numpy=True)[0].astype("float32")
        q_emb_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)

        if USE_FAISS and self.index is not None:
            q_search = np.expand_dims(q_emb_norm, axis=0).astype("float32")
            D, I = self.index.search(q_search, top_k)
            idxs = [int(i) for i in I[0] if i >= 0]
        else:
            idxs = self._numpy_search(q_emb_norm, top_k)

        results = []
        seen = set()

        for idx in idxs:
            txt = self.docs[idx]["text"]
            if txt not in seen:
                results.append(self.docs[idx])
                seen.add(txt)

        if not results:
            return "Not found in uploaded documents."

        context = "\n\n---\n\n".join(
            [f"[source: {r['source']}] {r['text']}" for r in results]
        )

        client = get_llm_client()
        if not client:
            return context[:2000]

        system_msg = {
            "role": "system",
            "content": (
                "You are GuidePro AI. Use only the document context. "
                "If answer missing, say 'Not found in uploaded documents.'"
            )
        }

        user_msg = {
            "role": "user",
            "content": f"QUESTION: {q}\n\nCONTEXT:\n{context}"
        }

        return generate_answer(client, [system_msg, user_msg])
