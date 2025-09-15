# Autoschema-Graph-Rag

An experimental implementation of the [AutoschemaKG framework](https://github.com/HKUST-KnowComp/AutoSchemaKG) for building a GraphRAG search agent with autonomous knowledge graph construction.

## Overview

This project implements a knowledge graph-based retrieval-augmented generation (RAG) system that automatically constructs knowledge graphs from unstructured text without requiring predefined schemas. Unlike traditional knowledge graph construction methods that need domain experts to define schemas upfront, this system uses large language models to simultaneously extract knowledge triples and induce comprehensive schemas directly from text.

## Key Features

- **Autonomous Schema Induction**: Automatically generates knowledge graph schemas without manual ontology design
- **Multi-Modal Knowledge Extraction**: Extracts entities, events, and their relationships from text
- **Scalable Processing**: Handles large document collections with batch processing
- **Graph-Based RAG**: Implements retrieval-augmented generation using the constructed knowledge graphs
- **Neo4j Integration**: Stores and queries knowledge graphs using Neo4j database
- **FAISS Vector Search**: Enables fast similarity search over graph embeddings
- **API Server**: Provides REST API endpoints for querying the knowledge graph

## Architecture

The system consists of several key components:

### 1. Knowledge Graph Construction (`Atlas-KG-Constructor.py`)
- **Triple Extraction**: Uses LLMs to extract entity-entity, entity-event, and event-event relationships
- **Concept Generation**: Automatically generates conceptual abstractions for entities and events
- **Embedding Computation**: Creates vector embeddings for all graph nodes using sentence transformers
- **FAISS Indexing**: Builds efficient vector indices for fast retrieval

### 2. API Server (`atlas_api_server.py`)
- **Graph Retrieval**: Implements LargeKGRetriever and LargeKGToGRetriever for different retrieval strategies
- **Neo4j Integration**: Connects to Neo4j database for graph storage and querying
- **REST Endpoints**: Provides HTTP API for querying the knowledge graph
- **Multi-Model Support**: Supports different LLM backends (Groq, OpenAI-compatible APIs)

### 3. Document Processing
- **Text Extraction**: Processes various document formats and converts to plain text
- **Batch Processing**: Handles large document collections efficiently
- **Metadata Preservation**: Maintains document source information and relationships

## Installation

1. Clone the repository:
```bash
git clone https://github.com/quinnmpaddock/Autoschema-Graph-Rag.git
cd Autoschema-Graph-Rag
```

2. Install dependencies:
```bash
pip install atlas-rag
# For NV-embed-v2 support:
pip install atlas-rag[nvembed]
```

3. Set up environment variables:
```bash
export GROQ_API_KEY="your_groq_api_key"
```

4. Install and configure Neo4j:
- Install Neo4j database
- Set up authentication (default: user="neo4j", password="admin2025")
- Create a database (default: "test-db4")

## Usage

### 1. Knowledge Graph Construction

Configure your input data and run the knowledge graph constructor:

```python
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator

# Configure processing
kg_extraction_config = ProcessingConfig(
    model_path="llama-3.3-70b-versatile",
    data_directory="your-input-directory",
    filename_pattern="docs",
    batch_size_triple=3,
    batch_size_concept=16,
    output_directory="output-directory",
    max_new_tokens=2048,
    max_workers=3,
    remove_doc_spaces=True
)

# Run extraction pipeline
kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)
kg_extractor.run_extraction()
kg_extractor.convert_json_to_csv()
kg_extractor.generate_concept_csv_temp()
kg_extractor.create_concept_csv()
```

### 2. Start the API Server

Launch the GraphRAG API server:

```bash
python atlas_api_server.py
```

The server will start on `http://0.0.0.0:10085` and provide endpoints for querying the knowledge graph.

### 3. Query the Knowledge Graph

Send queries to the API server to retrieve information from your knowledge graph:

```bash
curl -X POST http://localhost:10085/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the relationship between entity A and entity B?"}'
```

## Project Structure

```
├── Atlas-KG-Constructor.py          # Main KG construction script
├── atlas_api_server.py              # REST API server
├── graph_retrieval.py               # Graph-based retrieval logic
├── KG_importer.py                   # Neo4j import utilities
├── documentFetch.py                  # Document processing utilities
├── config.yaml                       # Configuration file
├── ATLAS-meeting-minutes-input/      # Sample input documents
├── import-TEST/ATLAS-docs/           # Processed output data
├── neo4j/                            # Neo4j database files
└── meeting-notes-plaintext/          # Processed text files
```

## Output

The system generates:
- **CSV Files**: Structured triples and concept mappings
- **Neo4j Dumps**: Graph database imports
- **FAISS Indices**: Vector search indices
- **Embeddings**: Node and text embeddings

## References

- [AutoSchemaKG Paper](https://arxiv.org/html/2505.23628v1)
- [HKUST-KnowComp/AutoSchemaKG](https://github.com/HKUST-KnowComp/AutoSchemaKG)
```

---
