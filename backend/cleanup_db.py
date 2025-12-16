"""
Clean up Neo4j and ChromaDB databases.
Run this script to remove all data and start fresh.
"""
import os
from app.services.graph_service import GraphService
import chromadb

def cleanup_neo4j():
    """Delete all nodes and relationships from Neo4j."""
    print("🧹 Cleaning up Neo4j...")
    graph_service = GraphService()
    try:
        # Delete all relationships first
        graph_service.driver.execute_query("MATCH ()-[r]->() DELETE r")
        print("  ✓ Deleted all relationships")
        
        # Delete all nodes
        graph_service.driver.execute_query("MATCH (n) DELETE n")
        print("  ✓ Deleted all nodes")
        
        graph_service.close()
        print("✅ Neo4j cleaned successfully")
    except Exception as e:
        print(f"❌ Error cleaning Neo4j: {e}")
        graph_service.close()

def cleanup_chromadb(collection_name: str = None):
    """Delete ChromaDB collection."""
    print("🧹 Cleaning up ChromaDB...")
    try:
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
        client = chromadb.PersistentClient(path=chroma_dir)
        
        if collection_name:
            # Delete specific collection
            try:
                client.delete_collection(name=collection_name)
                print(f"  ✓ Deleted collection: {collection_name}")
            except Exception as e:
                print(f"  ⚠ Collection '{collection_name}' not found or already deleted")
        else:
            # Delete all collections
            collections = client.list_collections()
            for collection in collections:
                client.delete_collection(name=collection.name)
                print(f"  ✓ Deleted collection: {collection.name}")
        
        print("✅ ChromaDB cleaned successfully")
    except Exception as e:
        print(f"❌ Error cleaning ChromaDB: {e}")

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("DATABASE CLEANUP SCRIPT")
    print("=" * 60)
    
    # Check if specific collection name provided
    collection_name = sys.argv[1] if len(sys.argv) > 1 else None
    
    if collection_name:
        print(f"\n⚠️  Will delete collection: {collection_name}")
        print("⚠️  Will delete ALL Neo4j data (entities & relationships)")
    else:
        print("\n⚠️  Will delete ALL ChromaDB collections")
        print("⚠️  Will delete ALL Neo4j data (entities & relationships)")
    
    response = input("\nAre you sure? (yes/no): ").strip().lower()
    
    if response == "yes":
        print("\n" + "=" * 60)
        cleanup_chromadb(collection_name)
        print()
        cleanup_neo4j()
        print("=" * 60)
        print("\n✅ Cleanup complete! You can now upload documents fresh.\n")
    else:
        print("\n❌ Cleanup cancelled.\n")
