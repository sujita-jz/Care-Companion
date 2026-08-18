"""
ChromaDB manager for medical knowledge base - Enhanced for RAG
- Supports original knowledge_base.json
- Supports PDF directory ingestion via RAG engine
- Provides retrieval with metadata, source citation
- Fallback in-memory if Chroma not available
"""
import os
import json

try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None
    CHROMA_AVAILABLE = False
    print("ChromaDB not installed, using in-memory fallback knowledge base")


class ChromaKnowledgeBase:
    def __init__(self, persist_dir="./chroma_storage", kb_file="knowledge_base.json"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.collection = None
        self.client = None
        self.fallback_docs = []  # in-memory if chroma missing, list of dicts with content, metadata, id

        if CHROMA_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=self.persist_dir)
            except Exception as e:
                print(f"Chroma persistent client failed {e}, using ephemeral")
                try:
                    self.client = chromadb.EphemeralClient()
                except Exception as e2:
                    print(f"Ephemeral also failed {e2}")
                    self.client = None

            if self.client:
                try:
                    self.collection = self.client.get_or_create_collection(
                        name="medical_knowledge",
                        metadata={"hnsw:space": "cosine"}
                    )
                except Exception as e:
                    print(f"Collection creation error {e}")
                    self.collection = None

        self.kb_file = kb_file
        self._initialized = False

    def initialize_if_empty(self):
        # Load from JSON for both chroma and fallback
        data = []
        if os.path.exists(self.kb_file):
            try:
                with open(self.kb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"KB file load error {e}")
        else:
            alt = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base.json")
            if os.path.exists(alt):
                self.kb_file = alt
                try:
                    with open(alt, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except:
                    data = []

        # Fallback memory - store as unified format
        if not self.fallback_docs and data:
            for item in data:
                self.fallback_docs.append({
                    "id": item['id'],
                    "content": item['content'],
                    "metadata": {"topic": item['topic'], "source": f"knowledge_base.json/{item['id']}",
                                 "id": item['id']}
                })

        if not self.collection:
            self._initialized = True
            return bool(self.fallback_docs)

        try:
            count = self.collection.count()
            if count > 0:
                self._initialized = True
                return True
        except:
            pass

        if data:
            try:
                documents = []
                metadatas = []
                ids = []
                for item in data:
                    # Avoid duplicate if already exists - check
                    documents.append(item['content'])
                    metadatas.append(
                        {"topic": item['topic'], "id": item['id'], "source": f"knowledge_base.json/{item['id']}",
                         "type": "json_kb"})
                    ids.append(item['id'])

                if documents:
                    # Use upsert to avoid duplicates
                    try:
                        self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                    except:
                        try:
                            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                        except Exception as inner:
                            print(f"Add json kb error (might be duplicates): {inner}")
                    print(f"ChromaKB: Added/Upserted {len(documents)} documents from JSON")
                    self._initialized = True
                    return True
            except Exception as e:
                print(f"ChromaKB init error: {e}")
                return False
        return False

    def search(self, query: str, n_results=5, where_filter=None) -> list:
        """Main retrieval for RAG - returns list of {content, metadata, distance}"""
        results_final = []
        if self.collection:
            try:
                self.initialize_if_empty()
                query_params = {
                    "query_texts": [query],
                    "n_results": n_results
                }
                if where_filter:
                    query_params["where"] = where_filter

                results = self.collection.query(**query_params)
                if results and 'documents' in results and results['documents']:
                    docs = results['documents'][0]
                    metas = results['metadatas'][0] if results.get('metadatas') and results['metadatas'][
                        0] else [{}] * len(docs)
                    distances = results['distances'][0] if results.get('distances') and results['distances'][
                        0] else [0] * len(docs)
                    ids = results['ids'][0] if results.get('ids') and results['ids'][0] else [None] * len(docs)
                    for i, doc in enumerate(docs):
                        results_final.append({
                            "content": doc,
                            "metadata": metas[i] if i < len(metas) else {},
                            "distance": distances[i] if i < len(distances) else 0,
                            "id": ids[i] if i < len(ids) else None,
                            "source": (metas[i].get("source") if i < len(metas) else "unknown")
                        })
                    # If Chroma returned something, return it
                    if results_final:
                        return results_final
            except Exception as e:
                print(f"Chroma search error: {e}")

        # Fallback simple keyword search over fallback_docs + also try to include any cached
        if not self.fallback_docs:
            self.initialize_if_empty()

        q_lower = query.lower()
        scored = []
        for item in self.fallback_docs:
            # item can be dict with content/metadata or old format
            content = item.get('content', '') if isinstance(item, dict) else str(item)
            meta = item.get('metadata', {}) if isinstance(item, dict) else {}
            content_lower = content.lower()
            topic = meta.get('topic', '') if isinstance(meta, dict) else ''
            source = meta.get('source', '') if isinstance(meta, dict) else ''
            topic_score = 0
            for word in q_lower.split():
                if len(word) < 3:
                    continue
                if word in content_lower:
                    topic_score += 1
                if word in topic.lower():
                    topic_score += 2
                if word in source.lower():
                    topic_score += 0.5
            if topic_score > 0 or not q_lower.strip():
                scored.append({
                    "content": content,
                    "metadata": meta,
                    "distance": 1.0 / (topic_score + 0.1),
                    "id": item.get('id'),
                    "source": source
                })
        scored.sort(key=lambda x: x['distance'])
        return scored[:n_results]

    def rag_retrieve(self, query: str, n_results=5, distance_threshold=1.2) -> dict:
        """
        RAG retrieval with threshold and context building
        Returns dict with:
          - context_text: concatenated context for LLM
          - sources: list of source citations
          - chunks: raw chunks
          - has_relevant: bool
        """
        chunks = self.search(query, n_results=n_results)

        # Filter by distance threshold
        relevant = [c for c in chunks if c['distance'] < distance_threshold] if chunks and isinstance(
            chunks[0].get('distance'), (int, float)) else chunks

        has_relevant = len(relevant) > 0
        # If fallback used, distance is inverted (lower is better still, but threshold 1.2 works)
        # For chroma cosine distance: 0=identical, 2=opposite. So threshold 0.8-1.2 is decent for relevant.
        # For fallback we already use 1/(score) so lower still better.

        context_parts = []
        sources = []
        for idx, chunk in enumerate(relevant if has_relevant else chunks[:2]):
            meta = chunk.get('metadata', {})
            source_display = meta.get('source_display') or meta.get('source') or meta.get('topic') or f"chunk_{idx}"
            page = meta.get('page', '')
            src_line = f"Source {idx + 1}: {source_display} {f'Page {page}' if page else ''}".strip()
            context_parts.append(f"[{src_line}]\n{chunk['content']}\n")
            sources.append({
                "source": source_display,
                "page": page,
                "topic": meta.get('topic', ''),
                "distance": chunk.get('distance'),
                "id": chunk.get('id')
            })

        context_text = "\n---\n".join(context_parts)

        return {
            "context_text": context_text,
            "sources": sources,
            "chunks": relevant if has_relevant else chunks,
            "has_relevant": has_relevant,
            "all_chunks": chunks
        }

    def add_document(self, doc_id, content, topic="general", metadata_extra=None):
        meta = {"topic": topic}
        if metadata_extra:
            meta.update(metadata_extra)
        if self.collection:
            try:
                self.collection.upsert(
                    documents=[content],
                    metadatas=[meta],
                    ids=[doc_id]
                )
                return True
            except Exception as e:
                try:
                    self.collection.add(documents=[content], metadatas=[meta], ids=[doc_id])
                    return True
                except Exception as e2:
                    print(f"Add doc error {e2}")
                    return False
        else:
            self.fallback_docs.append({"id": doc_id, "content": content, "metadata": meta})
            return True

    @property
    def count(self):
        if self.collection:
            try:
                return self.collection.count()
            except:
                return len(self.fallback_docs)
        return len(self.fallback_docs)

    def delete_by_source(self, source_value: str):
        """Delete all chunks where metadata source == source_value"""
        if self.collection:
            try:
                self.collection.delete(where={"source": source_value})
                return True
            except Exception as e:
                # Try alternative: where document filter may not be supported, try get ids then delete
                try:
                    # Query all with filter if possible
                    res = self.collection.get(where={"source": source_value})
                    ids = res.get('ids', [])
                    if ids:
                        self.collection.delete(ids=ids)
                        return True
                except Exception as e2:
                    print(f"Delete by source failed {e2}")
        else:
            # fallback: remove from list
            self.fallback_docs = [d for d in self.fallback_docs if d.get('metadata', {}).get('source') != source_value]
            return True
        return False


# Singleton
_kb_instance = None


def get_kb():
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = ChromaKnowledgeBase()
        _kb_instance.initialize_if_empty()
    return _kb_instance
