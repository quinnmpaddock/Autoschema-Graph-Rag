#!/bin/bash

NEO4J_VERSION=2025.03.0
NEO4J_BASE_DIR="$(pwd)"
SERVER="${NEO4J_BASE_DIR}/app-neo4j-community-${NEO4J_VERSION}"

if [ -d "$SERVER" ]; then
  echo "Stopping Neo4j server in $SERVER..."
  "$SERVER/bin/neo4j" stop
else
  echo "Directory $SERVER does not exist, skipping."
fi 
