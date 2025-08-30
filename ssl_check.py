# ssl_check.py
import os
import requests

print("--- SSL Certificate Check ---")

# 1. Check for the environment variable from within Python
ssl_cert_file = os.environ.get('SSL_CERT_FILE')
if ssl_cert_file:
    print(f"✅ Environment variable SSL_CERT_FILE is set to: {ssl_cert_file}")
    # 2. Check if the file actually exists
    if os.path.exists(ssl_cert_file):
        print(f"✅ The file at that path exists.")
    else:
        print(f"❌ ERROR: The file at {ssl_cert_file} does NOT exist.")
else:
    print("❌ ERROR: Environment variable SSL_CERT_FILE is NOT set.")

# 3. Attempt a secure connection
print("\nAttempting to connect to https://www.google.com...")
try:
    response = requests.get("https://www.google.com", timeout=10)
    response.raise_for_status()
    print("✅ SUCCESS: Secure connection to Google.com was successful.")
except requests.exceptions.SSLError as e:
    print("\n❌❌❌ CRITICAL FAILURE ❌❌❌")
    print("Failed due to an SSL Error. This is the root cause.")
    print(f"DETAILS: {e}")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")

print("\n--- End of Check ---")
