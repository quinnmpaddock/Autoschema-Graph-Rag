from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "admin2025"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

try:
    with driver.session() as session:
        # Show all indexes, filter for vector
        result = session.run("""
        SHOW INDEXES YIELD 
          name, type, labelsOrTypes, properties, 
          options
        WHERE type = 'VECTOR'
        RETURN name, labelsOrTypes, properties, options
        """)
        
        records = list(result)
        if records:
            print("✅ Vector indexes found:")
            for r in records:
                print(f"  Name: {r['name']}")
                print(f"  Label: {r['labelsOrTypes']}")
                print(f"  Property: {r['properties']}")
                print(f"  Options: {r['options']}")
                print("  ---")
        else:
            print("❌ No vector index found!")
            print("You must create one — likely on `Passage` or `Text` nodes with `embedding` property.")

finally:
    driver.close()
