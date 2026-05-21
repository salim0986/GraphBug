from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. SETUP
client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. THE USER QUERY
query_text = "Logic for calculating government fees on payment"
print(f"🔎 Searching for: '{query_text}'...")

# 3. EMBED
query_vector = model.encode(query_text).tolist()

# 4. SEARCH (Updated for v1.10+)
# We use 'query_points' instead of 'search'
search_result = client.query_points(
    collection_name="code_snippets",
    query=query_vector, # <--- Argument name is 'query', not 'query_vector'
    limit=1
).points # <--- Note: We must access .points to get the list

# 5. SHOW RESULTS
for hit in search_result:
    print(f"\n✅ FOUND MATCH: {hit.payload['name']}")
    print(f"   Score: {hit.score:.4f}")
    print(f"   Code:\n{hit.payload['raw_code']}")