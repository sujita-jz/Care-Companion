"""
Quick RAG test script for Care Companion
Demonstrates knowledge_base directory scanning + RAG retrieval
"""
from utils.chroma_manager import get_kb
from utils.rag_engine import get_rag_loader
from utils.llm_handler import get_llm_handler

print("=== Care Companion RAG Test ===\n")

kb = get_kb()
kb.initialize_if_empty()

loader = get_rag_loader(kb_dir='knowledge_base')
print(f"Scanning {loader.kb_dir}...")
stats = loader.get_stats()
print(f"Found {stats['total_files']} files on disk: {stats['files_list']}")

print("\nIngesting...")
report = loader.ingest_directory(kb, force_reingest=True)
print(
    f"Report: {report['discovered_files']} discovered, {report['processed_files']} processed, {report['total_chunks']} chunks")

print(f"\nChroma count: {kb.count}")

# Test queries
queries = [
    "What does knowledge base say about fever management?",
    "Explain diabetes type 2 management from knowledge base",
    "What are precautions for skin wound per knowledge base?",
    "How to manage hypertension?",
    "What is medication safety?"
]

llm = get_llm_handler()

for q in queries:
    print(f"\n--- Query: {q} ---")
    rag_res = kb.rag_retrieve(q, n_results=3)
    print(f"Has relevant: {rag_res['has_relevant']}, chunks: {len(rag_res['chunks'])}")
    for src in rag_res['sources'][:2]:
        print(f"  Source: {src['source']} dist={src['distance']:.3f}")

    # Generate response (will use fallback if no Gemini key)
    result = llm.generate_text_response(q, preferred_lang='en',
                                        user_profile={"age": 28, "health_conditions": "None", "allergies": "None"})
    print(f"Response source: {result['source']}")
    print(f"Response preview: {result['response'][:500]}...\n")
    if result.get('rag_sources'):
        print(f"RAG Sources cited: {[s['source'] for s in result['rag_sources'][:3]]}")

print("\n=== Test Done ===")
print("Add your own PDFs to knowledge_base/ and re-run: python test_rag.py")
