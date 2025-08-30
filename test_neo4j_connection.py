from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection details — same as in atlas_api_server.py
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "admin2025"

def test_neo4j_data():
    # Create a REAL Neo4j driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Test connection
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS total")
            count = result.single()["total"]
            logger.info(f"📊 Total nodes in DB: {count}")

            if count == 0:
                logger.warning("⚠️ No nodes found! Your graph may not have been populated.")
                return

            # Show labels
            labels_result = session.run("CALL db.labels() YIELD label")
            logger.info(f"🏷️  Labels in DB: {[r['label'] for r in labels_result]}")

            # Show sample nodes
            sample_result = session.run("""
                MATCH (n) 
                RETURN elementId(n) AS id, labels(n) AS labels, size(keys(n)) AS props 
                LIMIT 5
            """)
            logger.info("🔍 Sample nodes:")
            for record in sample_result:
                print(f"  ID: {record['id']}, Labels: {record['labels']}, Props: {record['props']}")

    except Exception as e:
        logger.error(f"💥 Failed to query Neo4j: {e}", exc_info=True)
    finally:
        driver.close()

if __name__ == "__main__":
    test_neo4j_data()
