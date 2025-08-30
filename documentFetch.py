import os
import re
import json
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Define the scopes your application needs access to.
SCOPES = [
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.readonly"
]

def main():
    """
    Fetches all documents from a specified Google Drive folder,
    extracts their content, and saves each as a JSON file.
    """
    folder_id = "1yvCa8qAFvchv2YjjTv6Z9OJ5AieusgtD"
    output_dir = "meeting-notes-json"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        docs_service, drive_service = authenticate()

        items = []
        page_token = None
        while True:
            # Find all Google Docs in the specified folder, ordered by creation time
            query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.document'"
            results = drive_service.files().list(
                q=query,
                fields="nextPageToken, files(id, name)",
                orderBy="createdTime",
                pageToken=page_token
            ).execute()
            
            items.extend(results.get('files', []))
            page_token = results.get('nextPageToken', None)
            if page_token is None:
                break

        if not items:
            print("No documents found in the specified folder.")
            return

        print(f"Found {len(items)} documents. Processing...")

        for i, item in enumerate(items):
            doc_id = item['id']
            doc_title = item['name']
            doc_numeric_id = i + 1

            # Sanitize filename and create the output path
            safe_filename = "".join(c for c in doc_title if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
            output_path = os.path.join(output_dir, f"{safe_filename}.json")

            # Skip if the file already exists
            if os.path.exists(output_path):
                print(f"Skipping '{doc_title}', as it already exists at '{output_path}'")
                continue

            try:
                # Retrieve the document using the Docs API.
                document = docs_service.documents().get(documentId=doc_id).execute()

                # Extract and print the text
                content = document.get('body', {}).get('content', [])
                lists = document.get('lists', {})
                extracted_text = extract_text_from_doc(content, lists)

                # Wrap the text in a JSON object
                json_output = create_json_object(extracted_text, doc_numeric_id)

                # Save the JSON
                with open(output_path, 'w') as f:
                    json.dump(json_output, f, indent=4)

                print(f"Successfully processed and saved '{doc_title}' to '{output_path}'")

            except HttpError as doc_err:
                print(f"An HTTP error occurred while processing document '{doc_title}' (ID: {doc_id}): {doc_err}")
            except Exception as doc_e:
                print(f"An unexpected error occurred while processing document '{doc_title}' (ID: {doc_id}): {doc_e}")

    except HttpError as err:
        print(f"An HTTP error occurred: {err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def authenticate():
    """Handles user authentication and returns API service objects for Docs and Drive."""
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    docs_service = build("docs", "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)
    return docs_service, drive_service


def extract_text_from_doc(content, lists):
    """
    Parses a list of content elements from a Google Doc to extract and
    clean plain text, converting titles, headings, and lists to Markdown format.
    """
    text = ""
    numbered_lists = {"DECIMAL", "UPPER_ALPHA", "LOWER_ALPHA", "UPPER_ROMAN", "LOWER_ROMAN"}

    for element in content:
        if 'paragraph' in element:
            paragraph = element.get('paragraph', {})
            para_elements = paragraph.get('elements', [])

            prefix = ""
            # Check for paragraph style to handle headings
            style = paragraph.get('paragraphStyle', {})
            named_style = style.get('namedStyleType')

            if named_style == 'TITLE':
                prefix = "# "
            elif named_style == 'HEADING_1':
                prefix = "## "
            elif named_style == 'HEADING_2':
                prefix = "### "
            elif named_style == 'HEADING_3':
                prefix = "#### "
            elif named_style == 'HEADING_4':
                prefix = "##### "
            elif named_style == 'HEADING_5':
                prefix = "###### "

            # Check for list items and handle nesting
            bullet = paragraph.get('bullet')
            if bullet:
                list_id = bullet.get('listId')
                nesting_level = bullet.get('nestingLevel', 0)
                indent = "  " * nesting_level

                effective_glyph_type = "GLYPH_TYPE_UNSPECIFIED"
                if list_id:
                    list_props = lists.get(list_id, {}).get('listProperties', {})
                    nesting_levels_props = list_props.get('nestingLevels', [])

                    if nesting_levels_props:
                        # Iterate from the current level up to the root to find the effective glyphType.
                        for i in range(min(nesting_level, len(nesting_levels_props) - 1), -1, -1):
                            level_props = nesting_levels_props[i]
                            if 'glyphType' in level_props:
                                effective_glyph_type = level_props['glyphType']
                                break

                if effective_glyph_type in numbered_lists:
                    prefix = f"{indent}1. " # Standard Markdown for numbered lists
                else:
                    prefix = f"{indent}- "

            text += prefix

            for para_element in para_elements:
                text_run = para_element.get('textRun')
                if text_run:
                    text += text_run.get('content', '')

        elif 'table' in element:
            table = element.get('table', {})
            for row in table.get('tableRows', []):
                row_text = []
                for cell in row.get('tableCells', []):
                    # Recursively extract text from within the cell
                    cell_text = extract_text_from_doc(cell.get('content', []), lists)
                    # Use a pipe '|' to separate cell content, like in Markdown tables
                    row_text.append(cell_text.strip())
                text += " | ".join(row_text) + "\n"

    return preprocess_text(text)

def preprocess_text(text):
    """Cleans the extracted text."""
    # Replace multiple newlines and spaces with a single one
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    # You could add other cleaning steps here, like fixing Unicode errors
    return text.strip()

def create_json_object(content, doc_numeric_id):
    """Wraps the given content in a JSON object."""
    obj = {
        "id": str(doc_numeric_id),
        "text": content,
        "metadata": {
            "lang": "en"
        }
    }
    return obj

if __name__ == "__main__":
    main()
