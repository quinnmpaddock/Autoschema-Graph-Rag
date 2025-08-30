import yaml
from neo4j import GraphDatabase

"""
KG_importer2.py 

Live importer that mirrors neo4j-admin database import wiring using :ID/:START_ID/:END_ID columns
for node/relationship identity, while ensuring FAISS downstream lookups work by storing
numeric_id as a STRING and indexing it.

Key behaviors:
- Nodes are MERGE'd using the CSV id_column/id_property (e.g., name:ID -> MERGE on name).
- :LABEL supports multiple labels separated by ';' and ignores blanks.
- numeric_id is stored as toString(row.numeric_id) and excluded from the generic props map.
- Relationship endpoints match the same identifiers referenced by :START_ID/:END_ID.
- One relationship per CSV row using apoc.create.relationship (no deduping).
- Uniqueness constraints on numeric_id are ensured for Node/Text/Concept labels.

Requirements:
- APOC plugin installed and enabled.
- CSV files must be accessible via Neo4j import dir (file:///...).
- config.yaml alongside this script defines connection and CSV lists.
"""


def _cypher_map_str(d: dict) -> str:
    """
    Turn a mapping of {prop_name: cypher_expression_string} into a Cypher map string.
    Expressions are expected to be valid Cypher snippets, inserted as-is.
    """
    if not d:
        return "{}"
    parts = []
    for k, v in d.items():
        parts.append(f"{k}: {v}")
    return "{ " + ", ".join(parts) + " }"




def import_nodes(session, node_file_info):
    """
    Import nodes from a single CSV file, preserving neo4j-admin identity wiring.

    - Use :LABEL to set dynamic labels (supports ';' separated list); ignore empty/blank labels.
    - Use id_column as identity value for MERGE against id_property (e.g., name:ID -> MERGE on name).
    - Do NOT store special columns (:LABEL and the id column itself) as properties.
    - Store numeric_id as a STRING (toString), matching FAISS mapping.
    - Store all other columns as-is (strings by default).
    """
    csv_path = node_file_info["path"]
    id_column = node_file_info["id_column"]  # e.g., "name:ID"
    id_property = node_file_info["id_property"]  # e.g., "name"
    label_column = node_file_info["label_column"]  # typically ":LABEL"

    print(f"  - Importing nodes from '{csv_path}'...")

    query = f"""
    LOAD CSV WITH HEADERS FROM 'file:///{csv_path}' AS row
    WITH row,
         [lbl IN (
           CASE WHEN row.`{label_column}` CONTAINS ';'
             THEN split(row.`{label_column}`, ';')
             ELSE [row.`{label_column}`]
           END
         ) WHERE trim(lbl) <> '' | trim(lbl)] AS labels
    WHERE size(labels) > 0 AND row.`{id_column}` IS NOT NULL AND row.`{id_column}` <> ''
    CALL apoc.merge.node(labels, {{ {id_property}: row.`{id_column}` }}) YIELD node
    WITH node, row
    // Store numeric_id as a string for FAISS lookups; do not cast to integer.
    SET node.numeric_id =
      CASE
        WHEN row.numeric_id IS NOT NULL AND trim(row.numeric_id) <> ''
          THEN toString(row.numeric_id)
        ELSE node.numeric_id
      END
    // Remove special columns so they do not get written into node props.
    WITH node, apoc.map.removeKeys(row, [':LABEL', $id_col, 'numeric_id']) AS props
    SET node += props
    RETURN count(node) AS processed_nodes
    """

    result = session.run(query, id_col=id_column)
    processed = result.single()["processed_nodes"]
    print(f"    ...done. Processed {processed} nodes from {csv_path}.")
    return processed


def import_edges(session, edge_file_info):
    """
    Import relationships from a single CSV file.

    - Create one relationship per row using apoc.create.relationship (no deduping),
      mirroring neo4j-admin multiplicity.
    - Endpoints are matched using the same identifiers referenced by :START_ID/:END_ID, via
      start_node_property/end_node_property configured in edge_file_info.
    - Relationship type is taken from rel_type_column (e.g., ":TYPE"). Blank/null rows are skipped.
    """
    csv_path = edge_file_info["path"]
    start_id_col = edge_file_info["start_id_column"]
    start_node_prop = edge_file_info["start_node_property"]
    end_id_col = edge_file_info["end_id_column"]
    end_node_prop = edge_file_info["end_node_property"]
    rel_type_col = edge_file_info["rel_type_column"]

    start_label_col = edge_file_info.get("start_label_column")
    end_label_col = edge_file_info.get("end_label_column")

    props_cfg = edge_file_info.get("properties", {})
    set_props = props_cfg.get("set", {}) or {}
    set_map = _cypher_map_str(set_props)

    # Build property maps and toggle for idempotent relationship merging by numeric_id
    set_props_no_id = {k: v for k, v in set_props.items() if k != 'numeric_id'}
    set_map_no_id = _cypher_map_str(set_props_no_id)
    use_merge = bool(edge_file_info.get("use_merge_on_rel_numeric_id", False))

    print(f"  - Importing edges from '{csv_path}'...")

    if start_label_col and end_label_col:
        # Use apoc.merge.node for labeled endpoints with dynamic labels from CSV
        if use_merge:
            query = f"""
            LOAD CSV WITH HEADERS FROM 'file:///{csv_path}' AS row
            WITH row,
                 [lbl IN (
                   CASE WHEN row.`{start_label_col}` CONTAINS ';'
                     THEN split(row.`{start_label_col}`, ';')
                     ELSE [row.`{start_label_col}`]
                   END
                 ) WHERE trim(lbl) <> '' | trim(lbl)] AS s_labels,
                 [lbl IN (
                   CASE WHEN row.`{end_label_col}` CONTAINS ';'
                     THEN split(row.`{end_label_col}`, ';')
                     ELSE [row.`{end_label_col}`]
                   END
                 ) WHERE trim(lbl) <> '' | trim(lbl)] AS e_labels
            WHERE size(s_labels) > 0 AND size(e_labels) > 0
            CALL apoc.merge.node(s_labels, {{ {start_node_prop}: trim(row.`{start_id_col}`) }}) YIELD node AS s
            CALL apoc.merge.node(e_labels, {{ {end_node_prop}:   trim(row.`{end_id_col}`)   }}) YIELD node AS e
            WITH s, e, row,
                 CASE WHEN row.`{rel_type_col}` IS NOT NULL AND trim(row.`{rel_type_col}`) <> ''
                      THEN trim(row.`{rel_type_col}`)
                      ELSE NULL END AS rel_type
            WHERE rel_type IS NOT NULL
            CALL apoc.merge.relationship(s, rel_type, {{numeric_id: toString(row.numeric_id)}}, {set_map_no_id}, e) YIELD rel
            RETURN count(rel) AS processed_relationships
            """
        else:
            query = f"""
            LOAD CSV WITH HEADERS FROM 'file:///{csv_path}' AS row
            WITH row,
                 [lbl IN (
                   CASE WHEN row.`{start_label_col}` CONTAINS ';'
                     THEN split(row.`{start_label_col}`, ';')
                     ELSE [row.`{start_label_col}`]
                   END
                 ) WHERE trim(lbl) <> '' | trim(lbl)] AS s_labels,
                 [lbl IN (
                   CASE WHEN row.`{end_label_col}` CONTAINS ';'
                     THEN split(row.`{end_label_col}`, ';')
                     ELSE [row.`{end_label_col}`]
                   END
                 ) WHERE trim(lbl) <> '' | trim(lbl)] AS e_labels
            WHERE size(s_labels) > 0 AND size(e_labels) > 0
            CALL apoc.merge.node(s_labels, {{ {start_node_prop}: trim(row.`{start_id_col}`) }}) YIELD node AS s
            CALL apoc.merge.node(e_labels, {{ {end_node_prop}:   trim(row.`{end_id_col}`)   }}) YIELD node AS e
            WITH s, e, row,
                 CASE WHEN row.`{rel_type_col}` IS NOT NULL AND trim(row.`{rel_type_col}`) <> ''
                      THEN trim(row.`{rel_type_col}`)
                      ELSE NULL END AS rel_type
            WHERE rel_type IS NOT NULL
            CALL apoc.create.relationship(s, rel_type, {set_map}, e) YIELD rel
            RETURN count(rel) AS processed_relationships
            """
    else:
        # Match endpoints by property only (unlabeled)
        if use_merge:
            query = f"""
            LOAD CSV WITH HEADERS FROM 'file:///{csv_path}' AS row
            MATCH (s {{ {start_node_prop}: trim(row.`{start_id_col}`) }})
            MATCH (e {{ {end_node_prop}:   trim(row.`{end_id_col}`)   }})
            WITH s, e, row,
                 CASE WHEN row.`{rel_type_col}` IS NOT NULL AND trim(row.`{rel_type_col}`) <> ''
                      THEN trim(row.`{rel_type_col}`)
                      ELSE NULL END AS rel_type
            WHERE rel_type IS NOT NULL
            CALL apoc.merge.relationship(s, rel_type, {{numeric_id: toString(row.numeric_id)}}, {set_map_no_id}, e) YIELD rel
            RETURN count(rel) AS processed_relationships
            """
        else:
            query = f"""
            LOAD CSV WITH HEADERS FROM 'file:///{csv_path}' AS row
            MATCH (s {{ {start_node_prop}: trim(row.`{start_id_col}`) }})
            MATCH (e {{ {end_node_prop}:   trim(row.`{end_id_col}`)   }})
            WITH s, e, row,
                 CASE WHEN row.`{rel_type_col}` IS NOT NULL AND trim(row.`{rel_type_col}`) <> ''
                      THEN trim(row.`{rel_type_col}`)
                      ELSE NULL END AS rel_type
            WHERE rel_type IS NOT NULL
            CALL apoc.create.relationship(s, rel_type, {set_map}, e) YIELD rel
            RETURN count(rel) AS processed_relationships
            """

    result = session.run(query)
    processed = result.single()["processed_relationships"]
    print(f"    ...done. Processed {processed} relationships from {csv_path}.")
    return processed


def main():
    print("Starting Neo4j data import process...")

    # Load config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: config.yaml not found. Please create it.")
        return

    neo4j_config = config["neo4j"]
    import_config = config["import_files"]

    # Connect to Neo4j
    try:
        driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["user"], neo4j_config["password"]),
        )
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")
    except Exception as e:
        print(f"ERROR: Could not connect to Neo4j. Please check your config.yaml settings. Details: {e}")
        return

    # Execute imports per file with isolation
    with driver.session(database=neo4j_config["database"]) as session:
       

        print("\n--- Phase 1: Importing Nodes ---")
        for node_file_info in import_config.get("nodes", []):
            try:
                import_nodes(session, node_file_info)
            except Exception as e:
                print(f"ERROR importing nodes from {node_file_info.get('path')}: {e}")

        print("\n--- Phase 2: Importing Edges ---")
        for edge_file_info in import_config.get("edges", []):
            try:
                import_edges(session, edge_file_info)
            except Exception as e:
                print(f"ERROR importing edges from {edge_file_info.get('path')}: {e}")

    driver.close()
    print("\nImport complete. All specified files have been processed.")


if __name__ == "__main__":
    main()
