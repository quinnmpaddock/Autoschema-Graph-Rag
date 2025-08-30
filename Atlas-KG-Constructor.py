import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI
from transformers import pipeline
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from atlas_rag.vectorstore.embedding_model import SentenceEmbedding
# from groq import Groq

loaded = load_dotenv()
print("dotenv loaded successfully?", loaded)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set! Please set it before running.")

# Load OpenRouter API key from config file
# config = ConfigParser()
#config.read('config.ini')
model_name = "llama-3.3-70b-versatile"
client = OpenAI(
  base_url = "https://api.groq.com/openai/v1/",
  api_key = groq_api_key,
)

# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# client = pipeline(
#     "text-generation",
#     model=model_name,
#     device_map="auto",
# )
filename_pattern = 'docs'
input_directory = 'ATLAS-meeting-minutes-input'
output_directory = f'import-Large-Test/ATLAS-{filename_pattern}'
triple_generator = LLMGenerator(client, model_name=model_name)

kg_extraction_config = ProcessingConfig(
      model_path = model_name,
      data_directory = input_directory,
      filename_pattern = filename_pattern,
      batch_size_triple = 3,
      batch_size_concept = 16,
      output_directory = f"{output_directory}",
      max_new_tokens = 2048,
    max_workers = 3,
      remove_doc_spaces=True, # For removing duplicated spaces in the document text
)
kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)

# construct entity&event graph
kg_extractor.run_extraction()

# Convert Triples Json to CSV
kg_extractor.convert_json_to_csv()

# Concept Generation
kg_extractor.generate_concept_csv_temp()
kg_extractor.create_concept_csv()

# Convert to Neo4j dumps
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_encoder = SentenceEmbedding(sentence_model)

# add numeric id to the csv so that we can use vector indices
kg_extractor.add_numeric_id()

# compute embedding
kg_extractor.compute_kg_embedding(sentence_encoder) # default encoder_model_name="all-MiniLM-L12-v2", only compute all embeddings except any concept related embeddings
# kg_extractor.compute_embedding(encoder_model_name="all-MiniLM-L12-v2")
# kg_extractor.compute_embedding(encoder_model_name="nvidia/NV-Embed-v2")

# create faiss index
kg_extractor.create_faiss_index() # default index_type="HNSW,Flat", other options: "IVF65536_HNSW32,Flat" for large KG
# kg_extractor.create_faiss_index(index_type="HNSW,Flat")
# kg_extractor.create_faiss_index(index_type="IVF65536_HNSW32,Flat")


