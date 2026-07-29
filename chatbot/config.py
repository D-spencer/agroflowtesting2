TOP_K = 3

RRF_SCORE_THRESHOLD = 0.10
MAX_HISTORY = 6

TOP_CONTEXT_CHUNKS = 2

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

LLM_MODEL = "llama-3.3-70b-versatile"


# LLM_MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.2

MAX_TOKENS = 350

ENABLE_RAG = True

ENABLE_CITATIONS = True

ENABLE_MEMORY = True

ENABLE_HYBRID_SEARCH = False

ENABLE_QUERY_REWRITE = False

ENABLE_RERANKING = False
