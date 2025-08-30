# Agentic Auto Schema Knowledge Graph Project

## 1. Use google docs API to retrieve documents
### a. Set up OAuth credentials in Google Cloud console
#### define correct scopes. for reading Google Docs, use https://www.googleapis.com/auth/documents.readonly scope. May also need https://www.googleapis.com/auth/drive.readonly to list and find documents
#### Download the credentials.json file and store it securely in project directory
### b. Write script to authenticate user
#### auth libraries: google-api-python-client, google-auth-oauthlib, and google-auth-httplib2
#### Implement Authentification flow where user is directed to Google consent screen to grant drive access to application. Request offline access to get access and refresh tokens
#### store refresh tokens in local file. Script should use stored refresh to request new acccess token on subsequent uses
#### set publishing status to "in production" to prevent refresh tokens from expiring quickly
### c. After authentification, use service,documents().get() to get json document from google
### d. Parse json to extract plaintext from json content blocks, ignoring formatting elements
#### Example simple formatting code:
```python
def extract_text_from_doc(doc_json):
    """Parses the JSON from the Google Docs API to extract plain text."""
    text = ""
    content = doc_json.get('body', {}).get('content', [])
    for element in content:
        if 'paragraph' in element:
            para_elements = element.get('paragraph', {}).get('elements', [])
            for para_element in para_elements:
                text_run = para_element.get('textRun')
                if text_run:
                    text += text_run.get('content', '')
    return text
```
#### enchance this code to better handle table rows and cells, bullet pointed lists, and headings
#### create "pre-processing" step to clean extracted text (eg. remove excessive newlines, fix unicode errors)

## 2. Feed text into autoschemaKG program
### a. Use scripts available on project git repo
#### set up the project environment to match requirements of autoschemaKG
### b. input plaintext extracted from json in prev step
#### ensure text is of the correct format and what is expected by autoschemaKG
### c. output triples 
#### output should be standard RDF format (eg. N-triples or turtle)
### d. Validate and review triples
#### write a script to sample a few dozen generated triples from a document
#### manually review triples for logical relationships and correctly identified entities
#### fine-tune autoschemaKG llm model/prompts or change preprocessing step if results are sub-par

## 3. Feed generated triples into Neo4j database
### a. install neo4j in venv (so entire app package can be easily transferred between computers)
### b. install neosemantics (n10s) in Neo4j and configure Neo4j to use plugin
### c. Use n10s to import RDF data (triples) into Neo4j database
#### Configure Neo4j database to use neosemantics
#### Use python script to loop through generated triples and add each one using n10s

## 4. Use lightRAG to query knowledge graph
### a. clone lightRAG repo and install dependencies
### b. Configure lightRAG to connect to neo4j database
#### enter Neo4j credentials: Bolt URL, username, and password
#### define data source for lightRAG to be Neo4j
#### test integration with sample questions
### c. Refine cypher query setup as needed 
#### Use vector similarity search
#### Cache recent queries/results or implement episodic memory (maybe?)
### d. Test and refine
#### begin with simple questions that map directly to single triples (eg. "What is the deadline for Some_paper?")
#### move to more complex questions
#### run automated benchmarks
## 5. Documentation and Usability
### a. create simple UI for querying and getting results
### b. Document pipeline, config options, and troubleshooting 

