import sys
import os

# This ensures the script can find the 'app' module by adding the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services.embeddings import get_embeddings

from langchain_community.vectorstores import Chroma

# This script must be run from the /app directory inside the container
# Example: docker exec hybridrag-backend python3 scripts/print_chunks.py

COLLECTION_NAME = "100_criminal_law_terms"
PERSIST_DIR = "./chroma_db"

print(f"--- Inspecting Chunks for Collection: {COLLECTION_NAME} ---")

try:
    # Initialize with the correct embedding function, same as the app
    embeddings = get_embeddings()

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    results = store._collection.get(include=["documents", "metadatas"])
    chunks = results.get('documents', [])
    
    if not chunks:
        print("\nERROR: No chunks found in the collection. The document may not have been ingested correctly.")
    else:
        print(f"\nFound {len(chunks)} chunks. Searching for 'White-Collar Crime'...")
        found = False
        for i, doc in enumerate(chunks):
            if 'white-collar' in doc.lower():
                print(f"\n{'='*10} FOUND IN CHUNK {i} {'='*10}")
                print(doc)
                found = True
        
        if not found:
            print("\nERROR: The term 'White-Collar Crime' was NOT found in any of the ingested chunks.")
            print("This confirms the issue is with PDF text extraction or the chunking process.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure the backend container is running and the collection name is correct.")

