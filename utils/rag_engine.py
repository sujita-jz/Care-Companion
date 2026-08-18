"""
RAG Engine for Care Companion
- Reads whole directory and subdirectories for PDFs, TXT, MD
- Chunks documents with overlap
- Ingests into ChromaDB with metadata (source file, page, chunk_id)
- Provides RAG retrieval + grounded generation
"""
import os
import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

# Text extraction helpers - optional deps
try:
    import PyPDF2

    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


class RAGKnowledgeLoader:
    def __init__(self, kb_dir: str = "knowledge_base", supported_exts=(".pdf", ".txt", ".md")):
        self.kb_dir = kb_dir
        os.makedirs(self.kb_dir, exist_ok=True)
        self.supported_exts = supported_exts
        self.ingestion_log = os.path.join(
            os.path.dirname(os.path.dirname(__file__)) if os.path.dirname(__file__) else ".", "ingestion_log.json")
        # Keep history of ingested files to avoid re-ingesting same hash unless changed
        self._load_log()

    def _load_log(self):
        if os.path.exists("ingestion_log.json"):
            self.ingestion_log = "ingestion_log.json"
        try:
            if os.path.exists(self.ingestion_log):
                with open(self.ingestion_log, 'r') as f:
                    self.log = json.load(f)
            else:
                self.log = {}
        except:
            self.log = {}

    def _save_log(self):
        try:
            with open(self.ingestion_log, 'w') as f:
                json.dump(self.log, f, indent=2)
        except Exception as e:
            print(f"Log save error {e}")

    def _file_hash(self, path: str) -> str:
        try:
            h = hashlib.md5()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            return h.hexdigest()
        except:
            return ""

    def discover_files(self) -> List[str]:
        """Recursively discover all supported files, excluding README and hidden"""
        files = []
        exclude_names = {"readme.md", "readme.txt", ".ds_store", "thumbs.db"}
        for root, dirs, filenames in os.walk(self.kb_dir):
            # Skip hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for fname in filenames:
                lower = fname.lower()
                if lower in exclude_names:
                    continue
                if lower.startswith('.'):
                    continue
                if lower.endswith(self.supported_exts):
                    full_path = os.path.join(root, fname)
                    if os.path.getsize(full_path) == 0:
                        continue
                    files.append(full_path)
        return sorted(files)

    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Returns list of (page_number, text)"""
        if not PYPDF_AVAILABLE:
            print(f"PyPDF2 not available, cannot read {pdf_path}")
            return []
        try:
            reader = PyPDF2.PdfReader(pdf_path)
            pages = []
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    # Clean
                    text = text.strip()
                    if text:
                        pages.append((i + 1, text))
                except Exception as e:
                    print(f"Page {i} extract error in {pdf_path}: {e}")
                    continue
            return pages
        except Exception as e:
            print(f"PDF read error {pdf_path}: {e}")
            return []

    def extract_text_from_txt(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except:
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                print(f"TXT read error {path}: {e}")
                return ""

    def extract_text_from_file(self, path: str) -> List[Dict]:
        """Unified extraction returns list of chunks dicts with metadata"""
        ext = Path(path).suffix.lower()
        source_rel = os.path.relpath(path, self.kb_dir)
        results = []

        if ext == ".pdf":
            pages = self.extract_text_from_pdf(path)
            for page_num, text in pages:
                results.append({
                    "text": text,
                    "metadata": {
                        "source": source_rel,
                        "full_path": path,
                        "page": page_num,
                        "type": "pdf"
                    }
                })
        else:  # txt, md
            text = self.extract_text_from_txt(path)
            if text.strip():
                results.append({
                    "text": text,
                    "metadata": {
                        "source": source_rel,
                        "full_path": path,
                        "page": 1,
                        "type": ext.lstrip(".")
                    }
                })
        return results

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 150, min_chunk: int = 100) -> List[str]:
        """
        Smart chunking:
        - First split by double newline (paragraphs)
        - Then maintain chunk_size with overlap
        - Keeps sentences intact where possible
        """
        if not text or len(text.strip()) < min_chunk:
            return [text] if text.strip() else []

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Split into sentences for better semantics
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current = ""
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > chunk_size and current:
                # Save current chunk
                chunks.append(current.strip())
                # Overlap: take last overlap chars from current for next
                overlap_text = current[-overlap:] if len(current) > overlap else current
                # Find sentence boundary in overlap
                current = overlap_text + " " + sent
                current_len = len(current)
            else:
                current = (current + " " + sent) if current else sent
                current_len += sent_len

        if current.strip():
            chunks.append(current.strip())

        # Filter very short
        final = []
        buf = ""
        for c in chunks:
            if len(c) < min_chunk:
                buf += " " + c
                if len(buf) >= min_chunk:
                    final.append(buf.strip())
                    buf = ""
            else:
                if buf:
                    final.append((buf + " " + c).strip())
                    buf = ""
                else:
                    final.append(c)
        if buf.strip():
            final.append(buf.strip())

        return final

    def prepare_chunks_for_ingestion(self, file_path: str) -> List[Dict]:
        """Extract + chunk a single file into list of {id, content, metadata}"""
        pieces = self.extract_text_from_file(file_path)
        all_chunks = []
        file_hash = self._file_hash(file_path)

        for piece_idx, piece in enumerate(pieces):
            raw_text = piece["text"]
            meta_base = piece["metadata"]
            text_chunks = self.chunk_text(raw_text, chunk_size=1000, overlap=150)

            for chunk_idx, chunk_text in enumerate(text_chunks):
                if not chunk_text.strip():
                    continue
                # Unique ID based on file hash + page + chunk index
                chunk_id = f"{file_hash[:8]}_{meta_base['page']}_{chunk_idx}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}"
                # Enrich metadata
                metadata = {
                    **meta_base,
                    "chunk_index": chunk_idx,
                    "file_hash": file_hash,
                    "chunk_id": chunk_id,
                    # For RAG citation
                    "source_display": f"{meta_base['source']} (p.{meta_base['page']})"
                }
                # Clean metadata values to be chroma friendly (no nested)
                clean_meta = {k: str(v)[:500] if isinstance(v, (str,)) and len(str(v)) > 500 else v for k, v in
                              metadata.items()}
                # Chroma doesn't like int keys? ensure strings
                all_chunks.append({
                    "id": chunk_id,
                    "content": chunk_text,
                    "metadata": clean_meta
                })
        return all_chunks

    def ingest_directory(self, chroma_kb, force_reingest: bool = False) -> Dict:
        """
        Full ingestion pipeline:
        - Discover files
        - Check hash vs log to skip unchanged
        - Chunk and add to ChromaDB
        - Returns report
        """
        files = self.discover_files()
        report = {
            "discovered_files": len(files),
            "processed_files": 0,
            "skipped_files": 0,
            "total_chunks": 0,
            "errors": [],
            "files": []
        }

        for fpath in files:
            try:
                fhash = self._file_hash(fpath)
                rel = os.path.relpath(fpath, self.kb_dir)

                # Check if already ingested and unchanged
                if not force_reingest and rel in self.log and self.log[rel].get("hash") == fhash:
                    report["skipped_files"] += 1
                    continue

                # If file changed, we should ideally delete old chunks first
                try:
                    if chroma_kb.collection:
                        chroma_kb.collection.delete(where={"source": rel})
                except Exception as e:
                    # Try alternative
                    try:
                        chroma_kb.delete_by_source(rel)
                    except:
                        pass

                chunks = self.prepare_chunks_for_ingestion(fpath)
                if not chunks:
                    report["skipped_files"] += 1
                    continue

                # Batch add to Chroma or fallback
                if chroma_kb.collection:
                    batch_size = 50
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        ids = [c["id"] for c in batch]
                        docs = [c["content"] for c in batch]
                        metas = [c["metadata"] for c in batch]
                        try:
                            chroma_kb.collection.add(ids=ids, documents=docs, metadatas=metas)
                        except Exception as e:
                            try:
                                chroma_kb.collection.upsert(ids=ids, documents=docs, metadatas=metas)
                            except Exception as e2:
                                print(f"Batch add error {fpath}: {e2}")
                                report["errors"].append(f"{rel}: {e2}")
                else:
                    # Fallback: add to in-memory kb
                    for c in chunks:
                        chroma_kb.fallback_docs.append({
                            "id": c["id"],
                            "content": c["content"],
                            "metadata": c["metadata"]
                        })

                # Update log
                self.log[rel] = {"hash": fhash, "chunks": len(chunks), "last_ingested": str(os.path.getmtime(fpath))}
                report["processed_files"] += 1
                report["total_chunks"] += len(chunks)
                report["files"].append({"file": rel, "chunks": len(chunks), "hash": fhash})

            except Exception as e:
                report["errors"].append(f"{fpath}: {str(e)}")
                print(f"Ingestion error {fpath}: {e}")

        self._save_log()
        return report

    def get_stats(self) -> Dict:
        files = self.discover_files()
        total_size = sum(os.path.getsize(f) for f in files)
        return {
            "kb_dir": self.kb_dir,
            "total_files": len(files),
            "total_size_bytes": total_size,
            "supported_exts": self.supported_exts,
            "files_list": [os.path.relpath(f, self.kb_dir) for f in files[:50]],  # first 50
            "ingestion_log_entries": len(self.log)
        }


# Singleton loader
_loader_instance = None


def get_rag_loader(kb_dir="knowledge_base"):
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = RAGKnowledgeLoader(kb_dir=kb_dir)
    return _loader_instance
