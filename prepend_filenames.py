import os
import sys

def prepend_docs_to_filenames(directory_path):
    """
    Prepends "docs_" to every file in the specified directory.

    Args:
        directory_path (str): The absolute path to the directory.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    print(f"Scanning directory: {directory_path}")

    try:
        for filename in os.listdir(directory_path):
            original_file_path = os.path.join(directory_path, filename)

            # Ensure we are only renaming files, not directories
            if os.path.isfile(original_file_path):
                # Check if the file already starts with "docs_" to avoid re-renaming
                if not filename.startswith("docs_"):
                    new_filename = f"docs_{filename}"
                    new_file_path = os.path.join(directory_path, new_filename)
                    
                    # Rename the file
                    os.rename(original_file_path, new_file_path)
                    print(f"Renamed: '{filename}' to '{new_filename}'")
                else:
                    print(f"Skipped (already prefixed): '{filename}'")

    except OSError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Check if a directory path is provided as a command-line argument
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        # If no argument is provided, ask the user for input
        target_directory = input("Please enter the absolute path to the directory: ")
    
    prepend_docs_to_filenames(target_directory)
