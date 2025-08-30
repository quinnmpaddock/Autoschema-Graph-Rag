import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from openai import OpenAI
from atlas_rag.llm_generator import LLMGenerator
from configparser import ConfigParser
from sentence_transformers import SentenceTransformer
from atlas_rag.vectorstore.embedding_model import SentenceEmbedding, NvEmbed
from neo4j import GraphDatabase
import faiss
import datetime
import logging
from logging.handlers import RotatingFileHandler
from atlas_rag.retriever.lkg_retriever.lkgr import LargeKGRetriever
from atlas_rag.retriever.lkg_retriever.tog import LargeKGToGRetriever
from atlas_rag.kg_construction.neo4j.neo4j_api import LargeKGConfig, start_app
from dotenv import load_dotenv

loaded = load_dotenv()
print("dotenv loaded successfully?", loaded)
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set! Please set it before running.")

# use sentence embedding if you want to use sentence transformer
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
sentence_encoder = SentenceEmbedding(sentence_model)
# Load OpenRouter API key from config file
#config = ConfigParser()
#config.read('config.ini')
# reader_model_name = "meta-llama/llama-3.3-70b-instruct"
retriever_model_name = "openai/gpt-oss-120b"
reader_model_name = "openai/gpt-oss-120b"
client = OpenAI(
  # base_url="https://openrouter.ai/api/v1",
  # api_key=config['settings']['OPENROUTER_API_KEY'],
  base_url="https://api.groq.com/openai/v1/",
  api_key= groq_api_key,
)
llm_generator = LLMGenerator(client=client, model_name=reader_model_name)

retriever_llm_generator = LLMGenerator(client=client, model_name=retriever_model_name)

# prepare necessary objects for instantiation of LargeKGRetriever: neo4j driver, faiss index etc.
neo4j_uri = "bolt://localhost:7687" # use bolt port for driver connection
user = "neo4j"
password = "admin2025"
database = "test-db4"
keyword = 'docs' # can be wiki or pes2o  # keyword to identify the cc_en dataset
driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

def check_neo4j_database_info(driver):
    """
    Simple checker to verify Neo4j connection and return database info
    """
    print("🔍 Checking Neo4j Database Connection...")
    print("=" * 50)
    
    try:
        with driver.session() as session:
            # Get current database name
            try:
                result = session.run("CALL db.info()")
                db_info = result.single()
                if db_info:
                    db_name = db_info.get('name', 'Unknown')
                    print(f"📊 Connected to database: '{db_name}'")
                else:
                    # Fallback method for older Neo4j versions
                    result = session.run("RETURN 'neo4j' as dbname")
                    db_name = result.single()['dbname']
                    print(f"📊 Connected to database: '{db_name}' (default)")
            except Exception as e:
                print(f"⚠️ Could not get database name: {e}")
                db_name = "Unknown"
            
            # Check node count
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()['node_count']
            print(f"📈 Total nodes in database: {node_count}")
            
            # Check node labels
            result = session.run("CALL db.labels()")
            labels = [record['label'] for record in result]
            print(f"🏷️ Node labels found: {labels if labels else 'None'}")
            
            # Check relationship types
            result = session.run("CALL db.relationshipTypes()")
            rel_types = [record['relationshipType'] for record in result]
            print(f"🔗 Relationship types: {rel_types if rel_types else 'None'}")
            
            # Check indexes
            try:
                result = session.run("SHOW INDEXES")
                indexes = list(result)
                print(f"📇 Indexes found: {len(indexes)}")
                for idx in indexes[:3]:  # Show first 3 indexes
                    idx_dict = dict(idx)
                    print(f"   - {idx_dict.get('name', 'Unknown')} ({idx_dict.get('type', 'Unknown')})")
                if len(indexes) > 3:
                    print(f"   ... and {len(indexes) - 3} more")
            except Exception as e:
                print(f"⚠️ Could not check indexes: {e}")
            
            print("✅ Database check completed!")
            return db_name, node_count
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None, 0

db_name, node_count = check_neo4j_database_info(driver)
print(f"\n🎯 API will use database: '{db_name}' with {node_count} nodes")

def debug_numeric_id_alignment(driver):
    with driver.session(database=database) as session:
        # Check numeric_id range in Neo4j
        result = session.run("MATCH (n:Node) WHERE n.numeric_id IS NOT NULL RETURN min(n.numeric_id) as min_id, max(n.numeric_id) as max_id, count(n) as count")
        stats = result.single()
        print(f"Neo4j Node numeric_ids: min={stats['min_id']}, max={stats['max_id']}, count={stats['count']}")
        
        # Check specific FAISS indices
        faiss_indices = [3, 34, 36, 10, 14, 5]
        for idx in faiss_indices:
            result = session.run("MATCH (n:Node) WHERE n.numeric_id = $idx RETURN n.name as name", idx=idx)
            node = result.single()
            if node:
                print(f"✅ FAISS index {idx} -> Neo4j node: {node['name']}")
            else:
                print(f"❌ FAISS index {idx} -> No matching Neo4j node")

# Add this to your atlas_api_server.py after the database check
debug_numeric_id_alignment(driver)

# Add the database check here
db_name, node_count = check_neo4j_database_info(driver)
print(f"\n🎯 API will use database: '{db_name}' with {node_count} nodes")

def load_faiss_index(index_path: str, index_name: str) -> faiss.Index:
    """Load FAISS index with existence and population check."""
    print(f"🔍 Checking FAISS index: {index_name}")
    print(f"   Path: {index_path}")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"❌ {index_name} not found at: {index_path}")

    try:
        index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
        print(f"✅ {index_name} loaded successfully")
        print(f"   Total vectors: {index.ntotal}")
        print(f"   Vector dimension: {index.d}")

        if index.ntotal == 0:
            raise ValueError(f"❌ {index_name} is empty (ntotal=0). No vectors to search.")

        return index

    except Exception as e:
        raise RuntimeError(f"❌ Failed to load {index_name}: {e}")


# Load indexes with validation
try:
    node_index_path = "import-Large-Test/ATLAS-docs/vector_index/triple_nodes_docs_from_json_with_emb_non_norm.index"
    text_index_path = "import-Large-Test/ATLAS-docs/vector_index/text_nodes_docs_from_json_with_emb_non_norm.index"

    node_index = load_faiss_index(node_index_path, "Node Index")
    text_index = load_faiss_index(text_index_path, "Text/Passage Index")

    print("🟢 All FAISS indexes loaded and valid. Proceeding with retriever setup...")

except (FileNotFoundError, ValueError, RuntimeError) as e:
    print(f"🔥 Critical error during index loading: {e}")
    print("🛑 Cannot start API — aborting.")
    exit(1)


# setup logger
date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
log_file_path = f'./log/LargeKGRAG_docs.log'
logger = logging.getLogger("LargeKGRAG")
logger.setLevel(logging.INFO)
max_bytes = 50 * 1024 * 1024  # 50 MB
if not os.path.exists(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
handler = RotatingFileHandler(log_file_path, maxBytes=max_bytes, backupCount=5)
logger.addHandler(handler)

retriever = LargeKGRetriever(keyword = keyword,
                             neo4j_driver=driver,
                             llm_generator=retriever_llm_generator,
                             sentence_encoder=sentence_encoder,
                             node_index= node_index,
                             passage_index=text_index,
                             topN = 5,
                             number_of_source_nodes_per_ner = 10,
                             sampling_area = 250,
                             logger = logger) # since cc_en is enormous compared to other dataset, we have different retrieval mechanism for it, which here we use keyword to identify cc_en.
tog_retriever = LargeKGToGRetriever(
    keyword = keyword,
    neo4j_driver=driver,
    topN = 5,
    Dmax = 2,
    Wmax = 3,
    llm_generator=retriever_llm_generator,
    sentence_encoder = sentence_encoder,
    filter_encoder = sentence_encoder,
    node_index = node_index,
    logger=logger
)

large_kg_config = LargeKGConfig(
    largekg_retriever = tog_retriever,
    reader_llm_generator = llm_generator, # you can use the same llm_generator as above or a different one for reading the retrieved passages,
    driver=driver,
    logger=logger,
    is_felm = False,
    is_mmlu = False,
    
)

start_app(user_config=large_kg_config, host="0.0.0.0", port = 10085, reload=False)
