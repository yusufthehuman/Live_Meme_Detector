import shutil
import os

if os.path.exists("dataset"):
    shutil.make_archive("dataset", "zip", "dataset")
    print("Successfully created 'dataset.zip'! Upload this file to Google Colab.")
else:
    print("Error: 'dataset' folder not found in your project directory.")