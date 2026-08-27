import os

def delete_every_third_file(folder_path):
    # Check if the directory exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Get a sorted list of all files in the folder (ignoring subdirectories)
    files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    files.sort()  # Sorting ensures a consistent order

    deleted_count = 0

    # Iterate through files; index starts at 0, so (i + 1) % 3 == 0 finds every 3rd file
    for i, file_name in enumerate(files):
        if (i + 1) % 3 == 0:
            file_path = os.path.join(folder_path, file_name)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_name}")
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {file_name}: {e}")

    print(f"\nFinished! Deleted {deleted_count} files out of {len(files)} total files.")

# --- Usage ---
# Replace with your actual folder path
folder_to_clean = r"dataset/temp"

delete_every_third_file(folder_to_clean)