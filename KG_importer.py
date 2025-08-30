#!/usr/bin/env python3
"""
Neo4j Knowledge Graph Importer - Configurable Version
- Fully configurable file names and locations via JSON config
- Incremental import with skip existing functionality
- Performance optimizations with connection pooling
- Proper error handling and validation
- Progress reporting and rollback capabilities
- Updated to handle actual CSV structure with all columns
"""

import os
import csv
import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple, Set
from pathlib import Path
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, ClientError, TransientError
import time
import json
from dataclasses import dataclass, field
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import ast

# ==== CONFIGURATION SYSTEM ====
@dataclass
class ImportConfig:
    """Type-safe configuration with file control."""
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    database_name: str
    batch_size: int
    max_transaction_retry_time: int
    connection_timeout: int
    incremental: bool = False
    max_workers: int = 4
    validate_files: bool = True
    create_constraints: bool = True
    clear_database: bool = False
    base_directory: str = "./import-TEST/ATLAS-docs"
    subdirectories: Dict[str, str] = field(default_factory=lambda: {
        "triples": "triples_csv",
        "concepts": "concept_csv"
    })
    file_mappings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    logging_config: Dict[str, str] = field(default_factory=lambda: {
        "level": "INFO",
        "file": "knowledge_graph_import.log",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    })
    constraints: Dict[str, Any] = field(default_factory=dict)

class DataTypeConverter:
    """Handles conversion of string representations to appropriate data types."""
  
    @staticmethod
    def convert_list_string(value: str) -> List[str]:
        """Convert string representation of list to actual list."""
        if not value or value.strip() == "[]":
            return []
        try:
            # Handle both string lists and actual lists
            if isinstance(value, str):
                # Remove extra whitespace and handle malformed lists
                value = value.strip()
                if value.startswith('[') and value.endswith(']'):
                    return ast.literal_eval(value)
                else:
                    # Handle comma-separated values without brackets
                    return [item.strip().strip("'\"") for item in value.split(',') if item.strip()]
            elif isinstance(value, list):
                return value
            else:
                return [str(value)]
        except (ValueError, SyntaxError):
            # If parsing fails, return as single-item list
            return [str(value)] if value else []
  
    @staticmethod
    def convert_property_value(key: str, value: Any) -> Any:
        """Convert property values based on key patterns."""
        if not value or (isinstance(value, str) and value.strip() == ""):
            return None
          
        # Handle list-like fields
        if key in ['concepts', 'synsets'] and isinstance(value, str):
            return DataTypeConverter.convert_list_string(value)
      
        # Handle numeric fields
        if key == 'numeric_id':
            try:
                return int(value) if value else None
            except (ValueError, TypeError):
                return None
      
        # Return as-is for other fields
        return value

class ConfigManager:
    """Configuration manager with full file control."""
  
    DEFAULT_CONFIG = {
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
            "database_name": "dulce-csv-json-text"
        },
        "settings": {
            "batch_size": 1000,
            "max_transaction_retry_time": 30,
            "connection_timeout": 10,
            "incremental": False,
            "max_workers": 4,
            "validate_files": True,
            "create_constraints": True,
            "clear_database": False
        },
        "file_locations": {
            "base_directory": "./import-TEST/ATLAS-docs",
            "subdirectories": {
                "triples": "triples_csv",
                "concepts": "concept_csv"
            }
        },
        "file_mappings": {
            "triple_nodes": {
                "filename": "triple_nodes_docs_from_json_without_emb_with_numeric_id.csv",
                "id_column": "name:ID",
                "label_column": ":LABEL",
                "expected_headers": ["name:ID", "type", "concepts", "synsets", "numeric_id", ":LABEL"],
                "property_columns": ["type", "concepts", "synsets", "numeric_id"],
                "enabled": True,
                "custom_path": None
            },
            "text_nodes": {
                "filename": "text_nodes_docs_from_json_with_numeric_id.csv",
                "id_column": "text_id:ID",
                "label_column": ":LABEL",
                "expected_headers": ["text_id:ID", "original_text", "numeric_id", ":LABEL"],
                "property_columns": ["original_text", "numeric_id"],
                "enabled": True,
                "custom_path": None
            },
            "concept_nodes": {
                "filename": "concept_nodes_docs_from_json_with_concept.csv",
                "id_column": "concept_id:ID",
                "label_column": ":LABEL",
                "expected_headers": ["concept_id:ID", "name", ":LABEL"],
                "property_columns": ["name"],
                "enabled": True,
                "custom_path": None
            },
            "triple_edges": {
                "filename": "triple_edges_docs_from_json_without_emb_with_numeric_id.csv",
                "expected_headers": [":START_ID", ":END_ID", "relation", "concepts", "synsets", "numeric_id", ":TYPE"],
                "property_columns": ["relation", "concepts", "synsets", "numeric_id"],
                "start_id_field": "name:ID",
                "end_id_field": "name:ID",
                "enabled": True,
                "custom_path": None
            },
            "text_edges": {
                "filename": "text_edges_docs_from_json.csv",
                "expected_headers": [":START_ID", ":END_ID", ":TYPE"],
                "property_columns": [],
                "start_id_field": "name:ID",
                "end_id_field": "text_id:ID",
                "enabled": True,
                "custom_path": None
            },
            "concept_edges": {
                "filename": "concept_edges_docs_from_json_with_concept.csv",
                "expected_headers": [":START_ID", ":END_ID", "relation", ":TYPE"],
                "property_columns": ["relation"],
                "start_id_field": "name:ID",
                "end_id_field": "concept_id:ID",
                "enabled": True,
                "custom_path": None
            }
        },
        "logging": {
            "level": "INFO",
            "file": "knowledge_graph_import.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "constraints": {
            "enabled": True,
            "definitions": [
                {
                    "type": "constraint",
                    "label": "Node",
                    "property": "name",
                    "statement": "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.name IS UNIQUE"
                },
                {
                    "type": "constraint", 
                    "label": "Text",
                    "property": "text_id",
                    "statement": "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Text) REQUIRE n.text_id IS UNIQUE"
                },
                {
                    "type": "constraint",
                    "label": "Concept", 
                    "property": "concept_id",
                    "statement": "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Concept) REQUIRE n.concept_id IS UNIQUE"
                },
                {
                    "type": "index",
                    "label": "Node",
                    "property": "name",
                    "statement": "CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.name)"
                },
                {
                    "type": "index",
                    "label": "Text", 
                    "property": "text_id",
                    "statement": "CREATE INDEX IF NOT EXISTS FOR (n:Text) ON (n.text_id)"
                },
                {
                    "type": "index",
                    "label": "Concept",
                    "property": "concept_id", 
                    "statement": "CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.concept_id)"
                }
            ]
        }
    }
  
    @classmethod
    def create_default_config(cls, config_path: str = "config.json"):
        """Create a default configuration file."""
        with open(config_path, 'w') as f:
            json.dump(cls.DEFAULT_CONFIG, f, indent=2)
        print(f"✅ Created configuration file: {config_path}")
  
    @classmethod
    def load_config(cls, config_path: str = "config.json") -> ImportConfig:
        """Load and validate configuration."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
          
            # Merge with defaults
            merged = cls.DEFAULT_CONFIG.copy()
            merged.update(config)
          
            return ImportConfig(
                neo4j_uri=merged["neo4j"]["uri"],
                neo4j_username=merged["neo4j"]["username"],
                neo4j_password=merged["neo4j"]["password"],
                database_name=merged["neo4j"]["database_name"],
                batch_size=merged["settings"]["batch_size"],
                max_transaction_retry_time=merged["settings"]["max_transaction_retry_time"],
                connection_timeout=merged["settings"]["connection_timeout"],
                incremental=merged["settings"].get("incremental", False),
                max_workers=merged["settings"].get("max_workers", 4),
                validate_files=merged["settings"].get("validate_files", True),
                create_constraints=merged["settings"].get("create_constraints", True),
                clear_database=merged["settings"].get("clear_database", False),
                base_directory=merged["file_locations"]["base_directory"],
                subdirectories=merged["file_locations"]["subdirectories"],
                file_mappings=merged["file_mappings"],
                logging_config=merged["logging"],
                constraints=merged["constraints"]
            )
          
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {config_path}")
            cls.create_default_config(config_path)
            return cls.load_config(config_path)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Invalid configuration: {e}")
            raise

class FilePathResolver:
    """Resolves file paths based on configuration."""
  
    def __init__(self, config: ImportConfig):
        self.config = config
        self.base_path = Path(config.base_directory)
  
    def get_file_path(self, file_key: str) -> Path:
        """Get the full path for a file based on configuration."""
        file_config = self.config.file_mappings.get(file_key)
        if not file_config or not file_config.get("enabled", True):
            return None
      
        # Check for custom path
        custom_path = file_config.get("custom_path")
        if custom_path:
            return Path(custom_path)
      
        # Build path from configuration
        filename = file_config["filename"]
      
        # Determine subdirectory
        if "triple" in file_key or "text" in file_key:
            subdir = self.config.subdirectories.get("triples", "triples_csv")
        elif "concept" in file_key:
            subdir = self.config.subdirectories.get("concepts", "concept_csv")
        else:
            subdir = ""
      
        return self.base_path / subdir / filename
  
    def get_all_file_paths(self) -> Dict[str, Path]:
        """Get all enabled file paths."""
        paths = {}
        for file_key in self.config.file_mappings.keys():
            path = self.get_file_path(file_key)
            if path:
                paths[file_key] = path
        return paths

# ==== CORE CLASSES ====
class ProgressTracker:
    """Tracks import progress across threads."""
  
    def __init__(self):
        self._lock = threading.Lock()
        self._processed = 0
        self._total = 0
  
    def set_total(self, total: int):
        """Set total items to process."""
        with self._lock:
            self._total = total
  
    def increment(self, count: int = 1):
        """Increment processed count."""
        with self._lock:
            self._processed += count
            if self._total > 0:
                percentage = (self._processed / self._total) * 100
                print(f"Progress: {self._processed}/{self._total} ({percentage:.1f}%)")

class CSVValidator:
    """Validates CSV files and headers."""
  
    @staticmethod
    def validate_file(file_path: Path, expected_headers: List[str]) -> Tuple[bool, str]:
        """Validate CSV file exists and has correct headers."""
        if not file_path:
            return False, "File not configured"
      
        if not file_path.exists():
            return False, f"File not found: {file_path}"
      
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                actual_headers = reader.fieldnames
              
                if not actual_headers:
                    return False, "File is empty"
              
                # Check if all expected headers are present (allow extra headers)
                missing = set(expected_headers) - set(actual_headers)
                if missing:
                    return False, f"Missing columns: {missing}"
              
                return True, f"Valid ({len(actual_headers)} columns)"
      
        except Exception as e:
            return False, f"Error reading file: {e}"

class CSVReader:
    """Handles CSV file reading with progress tracking."""
  
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
  
    def read_csv_in_batches(self, file_path: Path, progress_tracker: Optional[ProgressTracker] = None) -> Iterator[List[Dict[str, str]]]:
        try:
            # Count lines first (minus header)
            with open(file_path, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1

            if progress_tracker:
                progress_tracker.set_total(max(total_rows, 0))

            # Now read rows
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                batch = []
                for row in reader:
                    clean_row = {k: v for k, v in row.items() if v is not None}
                    batch.append(clean_row)
                    if len(batch) >= self.batch_size:
                        yield batch
                        if progress_tracker:
                            progress_tracker.increment(len(batch))
                        batch = []
                if batch:
                    yield batch
                    if progress_tracker:
                        progress_tracker.increment(len(batch))
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
            raise

class Neo4jConnectionManager:
    """Manages Neo4j connections with pooling and retry logic."""
  
    def __init__(self, config: ImportConfig):
        self.config = config
        self.driver = None
  
    def connect(self) -> None:
        """Connect to Neo4j with proper error handling."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_username, self.config.neo4j_password),
                max_transaction_retry_time=self.config.max_transaction_retry_time,
                connection_timeout=self.config.connection_timeout,
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            self.driver.verify_connectivity()
            print(f"Connected to Neo4j at {self.config.neo4j_uri}")
          
        except (ServiceUnavailable, AuthError) as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise
  
    def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            try:
                self.driver.close()
                print("Neo4j connection closed")
            except Exception as e:
                print(f"Error closing connection: {e}")
  
    @contextmanager
    def session(self):
        """Context manager for Neo4j sessions."""
        if not self.driver:
            raise RuntimeError("Database not connected")
      
        session = self.driver.session(database=self.config.database_name)
        try:
            yield session
        finally:
            session.close()

class NodeImporter:
    """Handles node imports with incremental support and proper property handling."""
  
    def __init__(self, connection_manager: Neo4jConnectionManager, config: ImportConfig):
        self.connection = connection_manager
        self.config = config
        self.csv_reader = CSVReader(config.batch_size)
        self.converter = DataTypeConverter()
  
    def _get_id_field_name(self, id_column: str) -> str:
        """Extract the actual field name from ID column specification."""
        return id_column.split(":")[0] if ":" in id_column else id_column
  
    def _check_existing_nodes(self, label: str, id_field: str, ids: List[str]) -> Set[str]:
        """Check which nodes already exist."""
        try:
            with self.connection.session() as session:
                result = session.run(
                    f"MATCH (n:`{label}`) WHERE n.{id_field} IN $ids RETURN n.{id_field} as id",
                    ids=ids
                )
                return {record["id"] for record in result}
        except Exception as e:
            print(f"Error checking existing nodes: {e}")
            return set()
  
    def import_nodes(self, file_config: Dict[str, Any], progress_tracker: Optional[ProgressTracker] = None) -> None:
        """Import nodes with incremental support and proper property handling."""
        file_path = file_config["path"]
        id_column = file_config["id_column"]
        label_column = file_config["label_column"]
        property_columns = file_config.get("property_columns", [])
        id_field = self._get_id_field_name(id_column)
      
        print(f"Starting node import from {file_path.name}")
      
        for batch in self.csv_reader.read_csv_in_batches(file_path, progress_tracker):
            if not batch:
                continue
          
            # Group by label and filter existing nodes if incremental
            nodes_by_label = {}
            for row in batch:
                label = row.get(label_column, "Node")
                if label not in nodes_by_label:
                    nodes_by_label[label] = []
              
                # Build properties from specified columns
                properties = {}
              
                # Add ID field
                if id_column in row and row[id_column]:
                    properties[id_field] = row[id_column]
              
                # Add property columns with type conversion
                for prop_col in property_columns:
                        if prop_col in row and row[prop_col]:
                            converted_value = self.converter.convert_property_value(prop_col, row[prop_col])
                        if converted_value is not None:
                            properties[prop_col] = converted_value
              
                # Add any additional columns not in property_columns but not system columns
                system_columns = {id_column, label_column}
                for col, val in row.items():
                    if col not in system_columns and col not in property_columns and val:
                        converted_value = self.converter.convert_property_value(col, val)
                        if converted_value is not None:
                            properties[col] = converted_value
              
                if properties:  # Only add if we have properties
                    nodes_by_label[label].append(properties)
          
            # Filter existing nodes if incremental
            if self.config.incremental:
                for label, nodes in nodes_by_label.items():
                    if nodes:
                        ids = [node.get(id_field) for node in nodes if node.get(id_field)]
                        if ids:
                            existing = self._check_existing_nodes(label, id_field, ids)
                            nodes_by_label[label] = [n for n in nodes if n.get(id_field) not in existing]
          
            # Import each label group
            for label, nodes in nodes_by_label.items():
                if not nodes:
                    continue
              
                query = f"""
                UNWIND $nodes AS node_data
                MERGE (n:`{label}` {{ {id_field}: node_data.{id_field} }})
                SET n += node_data
                """
              
                try:
                    with self.connection.session() as session:
                        session.run(query, nodes=nodes)
                    print(f"Created {len(nodes)} nodes with label '{label}'")
                  
                except ClientError as e:
                    if "already exists" in str(e).lower():
                        print(f"Skipping duplicate nodes for label '{label}': {e}")
                    else:
                        print(f"Error creating nodes with label '{label}': {e}")
                        raise
                except Exception as e:
                    print(f"Unexpected error creating nodes: {e}")
                    raise

class RelationshipImporter:
    """Handles relationship imports with optimized matching and proper property handling."""
  
    def __init__(self, connection_manager: Neo4jConnectionManager, config: ImportConfig):
        self.connection = connection_manager
        self.config = config
        self.csv_reader = CSVReader(config.batch_size)
        self.converter = DataTypeConverter()
  
    def _get_id_field_name(self, id_column: str) -> str:
        """Extract the actual field name from ID column specification."""
        return id_column.split(":")[0] if ":" in id_column else id_column
    def _label_for_id_field(self, field: str) -> Optional[str]:
        mapping = {
            "name": "Node",
            "text_id": "Text",
            "concept_id": "Concept",
        }
        return mapping.get(field)

    def _check_existing_relationships(
    self,
    rel_type: str,
    start_field: str,
    end_field: str,
    start_ids: List[str],
    end_ids: List[str],
    start_label: Optional[str],
    end_label: Optional[str],
) -> Set[Tuple[str, str]]:
        """Check which relationships already exist."""
        try:
            with self.connection.session() as session:
                start_label_clause = f":`{start_label}`" if start_label else ""
                end_label_clause = f":`{end_label}`" if end_label else ""
                query = f"""
                MATCH (start{start_label_clause})-[r:`{rel_type}`]->(end{end_label_clause})
                WHERE start.{start_field} IN $start_ids AND end.{end_field} IN $end_ids
                RETURN start.{start_field} as start_id, end.{end_field} as end_id
                """
                result = session.run(query, start_ids=start_ids, end_ids=end_ids)
                return {(record["start_id"], record["end_id"]) for record in result}
        except Exception as e:
            print(f"Error checking existing relationships: {e}")
            return set()

    def import_relationships(self, file_config: Dict[str, Any], progress_tracker: Optional[ProgressTracker] = None) -> None:
        """Import relationships with optimized matching and proper property handling."""
        file_path = file_config["path"]
        start_id_field = self._get_id_field_name(file_config["start_id_field"])
        end_id_field = self._get_id_field_name(file_config["end_id_field"])
        start_label = self._label_for_id_field(start_id_field)
        end_label = self._label_for_id_field(end_id_field)
        property_columns = file_config.get("property_columns", [])
      
        print(f"Starting relationship import from {file_path.name}")
      
        for batch in self.csv_reader.read_csv_in_batches(file_path, progress_tracker):
            relationships = []
          
            for row in batch:
                start_id = row.get(":START_ID")
                end_id = row.get(":END_ID")
                rel_type = row.get(":TYPE")
              
                if not all([start_id, end_id, rel_type]):
                    print(f"Skipping relationship with missing data: {row}")
                    continue
              
                # Build properties from specified columns
                properties = {}
              
                # Add property columns with type conversion
                for prop_col in property_columns:
                    if prop_col in row and row[prop_col]:
                        converted_value = self.converter.convert_property_value(prop_col, row[prop_col])
                        if converted_value is not None:
                            properties[prop_col] = converted_value
              
                # Add any additional columns not in property_columns but not system columns
                system_columns = {":START_ID", ":END_ID", ":TYPE"}
                for col, val in row.items():
                    if col not in system_columns and col not in property_columns and val:
                        converted_value = self.converter.convert_property_value(col, val)
                        if converted_value is not None:
                            properties[col] = converted_value
              
                relationships.append({
                    "start_id": start_id,
                    "end_id": end_id,
                    "type": rel_type,
                    "properties": properties
                })
          
            if relationships:
                # Filter existing relationships if incremental
                if self.config.incremental:
                    rel_type = relationships[0]["type"]
                    start_ids = [r["start_id"] for r in relationships]
                    end_ids = [r["end_id"] for r in relationships]
                    existing = self._check_existing_relationships(
                        rel_type,
                        start_id_field,
                        end_id_field,
                        start_ids,
                        end_ids,
                        start_label,
                        end_label,
                    )
                    relationships = [r for r in relationships if (r["start_id"], r["end_id"]) not in existing]
              
                # Import relationships in batches
                for rel in relationships:
                      query = f"""
                      MATCH (start:`{start_label}`) WHERE start.{start_id_field} = $start_id
                      MATCH (end:`{end_label}`) WHERE end.{end_id_field} = $end_id
                      MERGE (start)-[r:`{rel['type']}`]->(end)
                      SET r += $properties
                      """
                      try:
                          with self.connection.session() as session:
                              session.run(query, {
                                  "start_id": rel["start_id"],
                                  "end_id": rel["end_id"],
                                  "properties": rel["properties"]
                                })
                          
                      except Exception as e:
                          print(f"Error creating relationship {rel['type']} from {rel['start_id']} to {rel['end_id']}: {e}")
                      # Continue with other relationships instead of failing completely
                      continue
              
                print(f"Created {len(relationships)} relationships")

# ==== MAIN IMPORTER CLASS ====
class AdminCompatibleImporter:
    """Main importer class with full configuration control."""
  
    def __init__(self, config: ImportConfig):
        self.config = config
        self.connection = Neo4jConnectionManager(config)
        self.node_importer = NodeImporter(self.connection, config)
        self.rel_importer = RelationshipImporter(self.connection, config)
        self.progress_tracker = ProgressTracker()
        self.file_resolver = FilePathResolver(config)
      
        # Configure logging based on configuration
        self._setup_logging()
  
    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_config = self.config.logging_config
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            handlers=[
                logging.FileHandler(log_config.get("file", "knowledge_graph_import.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
  
    def validate_files(self) -> bool:
        """Validate all enabled CSV files exist and have correct structure."""
        if not self.config.validate_files:
            self.logger.info("File validation disabled in configuration")
            return True
      
        all_valid = True
        file_paths = self.file_resolver.get_all_file_paths()
      
        for file_key, file_path in file_paths.items():
            file_config = self.config.file_mappings[file_key]
            expected_headers = file_config["expected_headers"]
          
            is_valid, message = CSVValidator.validate_file(file_path, expected_headers)
            status = "✓" if is_valid else "✗"
            self.logger.info(f"{status} {file_key}: {file_path.name} - {message}")
            if not is_valid:
                all_valid = False
      
        return all_valid
  
    def get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics."""
        try:
            with self.connection.session() as session:
                result = session.run("""
                MATCH (n) 
                RETURN count(n) as nodes, 
                       count{(n)-[]-()} as relationships
                """)
                stats = result.single()
              
                # Get label counts
                labels_result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count")
                label_counts = {record["label"]: record["count"] for record in labels_result}
              
                # Get relationship type counts
                rels_result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
                rel_counts = {record["type"]: record["count"] for record in rels_result}
              
                return {
                    "nodes": stats["nodes"],
                    "relationships": stats["relationships"],
                    "labels": label_counts,
                    "relationship_types": rel_counts
                }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {}
  
    def create_constraints_and_indexes(self) -> None:
        """Create constraints and indexes based on configuration."""
        if not self.config.create_constraints:
            self.logger.info("Constraint creation disabled in configuration")
            return
      
        constraints_config = self.config.constraints
        if not constraints_config.get("enabled", True):
            self.logger.info("Constraints disabled in configuration")
            return
      
        try:
            with self.connection.session() as session:
                for definition in constraints_config.get("definitions", []):
                    try:
                        session.run(definition["statement"])
                        self.logger.debug(f"Created: {definition['statement']}")
                    except ClientError as e:
                        if "already exists" in str(e).lower():
                            self.logger.debug(f"Already exists: {definition['statement']}")
                        else:
                            raise
              
                session.run("CALL db.awaitIndexes()")
                self.logger.info("All constraints and indexes ready")
              
        except Exception as e:
            self.logger.error(f"Error creating constraints: {e}")
            raise
  
    def clear_database(self) -> None:
        """Clear all data from database."""
        if not self.config.clear_database:
            self.logger.info("Database clearing disabled in configuration")
            return
      
        try:
            with self.connection.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = result.single()["count"]
              
                if node_count > 0:
                    session.run("MATCH (n) DETACH DELETE n")
                    self.logger.info(f"Cleared {node_count} nodes from database")
                else:
                    self.logger.info("Database already empty")
                  
        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            raise
  
    def run_import(self) -> None:
        """Run complete import process with full configuration control."""
        start_time = time.time()
      
        try:
            # Pre-import validation
            if not self.validate_files():
                raise ValueError("Some CSV files are invalid or missing")
          
            # Get initial stats
            self.connection.connect()
            initial_stats = self.get_database_stats()
            self.logger.info(f"Initial database stats: {initial_stats}")
          
            # Clear database if configured
            self.clear_database()
          
            # Create constraints and indexes
            self.create_constraints_and_indexes()
          
            # Import nodes using dynamic file resolution
            self.logger.info("Starting node imports...")
            node_configs = ["triple_nodes", "text_nodes", "concept_nodes"]
          
            for node_key in node_configs:
                file_path = self.file_resolver.get_file_path(node_key)
                if file_path and file_path.exists():
                    file_config = self.config.file_mappings[node_key]
                    # Convert to expected format for backward compatibility
                    config_dict = {
                        "path": file_path,
                        "id_column": file_config["id_column"],
                        "label_column": file_config["label_column"],
                        "expected_headers": file_config["expected_headers"],
                        "property_columns": file_config.get("property_columns", [])
                    }
                    self.node_importer.import_nodes(config_dict, self.progress_tracker)
          
            # Import relationships using dynamic file resolution
            self.logger.info("Starting relationship imports...")
            edge_configs = ["triple_edges", "text_edges", "concept_edges"]
          
            for edge_key in edge_configs:
                file_path = self.file_resolver.get_file_path(edge_key)
                if file_path and file_path.exists():
                    file_config = self.config.file_mappings[edge_key]
                    # Convert to expected format for backward compatibility
                    config_dict = {
                        "path": file_path,
                        "expected_headers": file_config["expected_headers"],
                        "start_id_field": file_config["start_id_field"],
                        "end_id_field": file_config["end_id_field"],
                        "property_columns": file_config.get("property_columns", [])
                    }
                    self.rel_importer.import_relationships(config_dict, self.progress_tracker)
          
            # Final stats
            final_stats = self.get_database_stats()
            elapsed = time.time() - start_time
          
            self.logger.info(f"Import completed in {elapsed:.2f}s")
            self.logger.info(f"Final stats: {json.dumps(final_stats, indent=2)}")
          
            # Report changes
            if self.config.incremental and initial_stats:
                new_nodes = final_stats.get("nodes", 0) - initial_stats.get("nodes", 0)
                new_rels = final_stats.get("relationships", 0) - initial_stats.get("relationships", 0)
                self.logger.info(f"Added {new_nodes} new nodes and {new_rels} new relationships")
          
        except Exception as e:
            self.logger.error(f"Import failed: {e}")
            raise
        finally:
            self.connection.close()

def main():
    """Main function with comprehensive argument handling."""
    import argparse
  
    parser = argparse.ArgumentParser(description="Neo4j Knowledge Graph Importer")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--create-config", action="store_true", help="Create default config")
    parser.add_argument("--validate-only", action="store_true", help="Only validate files without importing")
    parser.add_argument("--list-files", action="store_true", help="List all configured files and their paths")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
  
    args = parser.parse_args()
  
    if args.create_config:
        ConfigManager.create_default_config(args.config)
        return
  
    try:
        config = ConfigManager.load_config(args.config)
    except Exception as e:
        print(f"❌ Config error: {e}")
        return
  
    importer = AdminCompatibleImporter(config)
  
    if args.list_files:
        file_paths = importer.file_resolver.get_all_file_paths()
        print("\\n📁 Configured file paths:")
        for key, path in file_paths.items():
            exists = "✅" if path.exists() else "❌"
            print(f"   {exists} {key}: {path}")
        return
  
    if args.validate_only:
        if importer.validate_files():
            print("✅ All files validated successfully")
        else:
            print("❌ Some files failed validation")
        return
  
    try:
        importer.run_import()
        print("✅ Import completed successfully!")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
