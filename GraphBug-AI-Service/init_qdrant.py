#!/usr/bin/env python3
"""
Initialize Qdrant Cloud cluster with required collections.
Run this script after creating your Qdrant Cloud cluster.
"""

import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def initialize_qdrant(url: str, api_key: str = None):
    """Create collections for code search."""
    
    print("🔗 Connecting to Qdrant...")
    
    if api_key:
        client = QdrantClient(url=url, api_key=api_key)
        print("✅ Connected with API key authentication")
    else:
        client = QdrantClient(url=url)
        print("✅ Connected without authentication (local)")
    
    try:
        # Check if collection exists
        collections = client.get_collections()
        existing_names = [c.name for c in collections.collections]
        
        print(f"\n📊 Existing collections: {existing_names if existing_names else 'None'}\n")
        
        collection_name = "repo_code"
        
        if collection_name in existing_names:
            print(f"⚠️  Collection '{collection_name}' already exists!")
            response = input("Do you want to recreate it? (yes/no): ")
            if response.lower() == 'yes':
                client.delete_collection(collection_name)
                print(f"🗑️  Deleted existing collection '{collection_name}'")
            else:
                print("✅ Keeping existing collection")
                return True
        
        # Create collection
        print(f"📦 Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,  # all-MiniLM-L6-v2 embedding size
                distance=Distance.COSINE
            )
        )
        print(f"✅ Collection '{collection_name}' created successfully!")
        
        # Show collection info
        info = client.get_collection(collection_name)
        print(f"\n📈 Collection Info:")
        print(f"   Name: {collection_name}")
        print(f"   Vector Size: {info.config.params.vectors.size}")
        print(f"   Distance: {info.config.params.vectors.distance}")
        print(f"   Points Count: {info.points_count}")
        
        print("\n🎉 Qdrant initialization complete!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Get credentials from environment or command line
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if len(sys.argv) >= 2:
        url = sys.argv[1]
    if len(sys.argv) >= 3:
        api_key = sys.argv[2]
    
    if not url:
        print("❌ Missing Qdrant URL!")
        print("\nUsage:")
        print("  python init_qdrant.py <url> [api_key]")
        print("\nOr set environment variables:")
        print("  export QDRANT_URL='https://xxxxx.aws.cloud.qdrant.io'")
        print("  export QDRANT_API_KEY='your-api-key'  # Optional for cloud")
        print("  python init_qdrant.py")
        sys.exit(1)
    
    print("=" * 60)
    print("🚀 Qdrant Cloud Initialization Script")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"API Key: {'***' + api_key[-8:] if api_key else 'None (local mode)'}")
    print("=" * 60)
    
    success = initialize_qdrant(url, api_key)
    sys.exit(0 if success else 1)
