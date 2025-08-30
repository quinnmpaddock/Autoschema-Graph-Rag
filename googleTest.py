# googleTest.py
import os
import logging
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.exceptions import GoogleAuthError
from oauthlib.oauth2.rfc6749.errors import OAuth2Error

# --- Enable Full Debug Logging ---
# This will print detailed information from the libraries we are using.
logging.basicConfig(level=logging.DEBUG)
# You can comment out specific loggers if the output is too noisy,
# but for now, let's see everything.
logging.getLogger('google_auth_oauthlib').setLevel(logging.DEBUG)
logging.getLogger('requests_oauthlib').setLevel(logging.DEBUG)
logging.getLogger('oauthlib').setLevel(logging.DEBUG)
logging.getLogger('urllib3').setLevel(logging.DEBUG)
# --- End of Logging Setup ---

# Use only one scope
SCOPES = [
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/documents.readonly'
]

def test():
    # Make sure we're using the right file
    if not os.path.exists('credentials.json'):
        print("credentials.json not found!")
        return
   
    flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            
    try:
        print("Starting local server and opening browser for authentication...")
        creds = flow.run_local_server(port=0)
        print("Authentication successful!")
        # The 'creds' object is now available for use.
        # For example, you could build a service object with it:
        # from googleapiclient.discovery import build
        # service = build('docs', 'v1', credentials=creds)
        # print("Service object created successfully.")

    except (GoogleAuthError, OAuth2Error, Exception) as e:
        print("\n--- An error occurred during the authentication flow ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("------------------------------------------------------")
        import traceback
        traceback.print_exc()
    

if __name__ == "__main__":
    test()
