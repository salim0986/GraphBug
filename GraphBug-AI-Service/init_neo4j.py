#!/usr/bin/env python3
"""
Initialize Neo4j Aura database with required indexes and constraints.
Run this script after creating your Neo4j Aura instance.
"""

import os
import sys
from neo4j import GraphDatabase

def initialize_neo4j(uri: str, user: str, password: str):
    """Create indexes and constraints for optimal performance."""
    
    print("🔗 Connecting to Neo4j...")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        # Verify connection
        driver.verify_connectivity()
        print("✅ Connected to Neo4j successfully!")
        
        with driver.session() as session:
            print("\n📊 Creating indexes and constraints...\n")
            
            # Indexes for fast lookups
            indexes = [
                # Repository index
                ("CREATE INDEX repo_id_index IF NOT EXISTS FOR (r:Repository) ON (r.repo_id)", 
                 "Repository.repo_id index"),
                
                # File indexes
                ("CREATE INDEX file_path_index IF NOT EXISTS FOR (f:File) ON (f.path)", 
                 "File.path index"),
                ("CREATE INDEX file_repo_index IF NOT EXISTS FOR (f:File) ON (f.repo_id)", 
                 "File.repo_id index"),
                
                # Class indexes
                ("CREATE INDEX class_name_index IF NOT EXISTS FOR (c:Class) ON (c.name)", 
                 "Class.name index"),
                ("CREATE INDEX class_file_index IF NOT EXISTS FOR (c:Class) ON (c.file)", 
                 "Class.file index"),
                
                # Function indexes
                ("CREATE INDEX function_name_index IF NOT EXISTS FOR (f:Function) ON (f.name)", 
                 "Function.name index"),
                ("CREATE INDEX function_file_index IF NOT EXISTS FOR (f:Function) ON (f.file)", 
                 "Function.file index"),
                
                # Variable index
                ("CREATE INDEX variable_name_index IF NOT EXISTS FOR (v:Variable) ON (v.name)", 
                 "Variable.name index"),
            ]
            
            for query, description in indexes:
                try:
                    session.run(query)
                    print(f"  ✅ {description}")
                except Exception as e:
                    print(f"  ⚠️  {description} - {e}")
            
            # Constraints (unique properties)
            constraints = [
                ("CREATE CONSTRAINT repo_id_unique IF NOT EXISTS FOR (r:Repository) REQUIRE r.repo_id IS UNIQUE",
                 "Repository.repo_id unique constraint"),
            ]
            
            for query, description in constraints:
                try:
                    session.run(query)
                    print(f"  ✅ {description}")
                except Exception as e:
                    print(f"  ⚠️  {description} - {e}")
            
            print("\n🎉 Neo4j initialization complete!")
            print("\n📈 Database Statistics:")
            
            # Show stats
            result = session.run("""
                CALL apoc.meta.stats() YIELD labels, relTypesCount
                RETURN labels, relTypesCount
            """)
            
            # Fallback if APOC not available
            try:
                stats = result.single()
                print(f"   Labels: {stats['labels']}")
                print(f"   Relationship Types: {stats['relTypesCount']}")
            except:
                # Simple count
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                count = result.single()["node_count"]
                print(f"   Total Nodes: {count}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    finally:
        driver.close()
    
    return True


if __name__ == "__main__":
    # Get credentials from environment or command line
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if len(sys.argv) == 4:
        uri = sys.argv[1]
        user = sys.argv[2]
        password = sys.argv[3]
    
    if not all([uri, user, password]):
        print("❌ Missing credentials!")
        print("\nUsage:")
        print("  python init_neo4j.py <uri> <user> <password>")
        print("\nOr set environment variables:")
        print("  export NEO4J_URI='neo4j+s://xxxxx.databases.neo4j.io'")
        print("  export NEO4J_USER='neo4j'")
        print("  export NEO4J_PASSWORD='your-password'")
        print("  python init_neo4j.py")
        sys.exit(1)
    
    print("=" * 60)
    print("🚀 Neo4j Aura Initialization Script")
    print("=" * 60)
    print(f"URI: {uri}")
    print(f"User: {user}")
    print("=" * 60)
    
    success = initialize_neo4j(uri, user, password)
    sys.exit(0 if success else 1)
