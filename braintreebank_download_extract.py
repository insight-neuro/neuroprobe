"""
    This script downloads all the files from the braintreebank.dev website.
    All the files are downloaded to the braintreebank/ directory.
"""
# Joined the two scripts into one. After the above script finishes downloading, the second script will run.
"""
Extract zip files in braintreebank_zip directory to braintreebank directory
"""

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import zipfile
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Download and extract files from braintreebank.dev')
parser.add_argument('--lite', action='store_true', 
                    help='Only download specific sub_ files: that are needed for the Neuroprobe benchmark. This option will reduce the number of files by >50%')
args = parser.parse_args()

# Define lite mode file list
NEUROPROBE_LITE_SUBJECT_TRIALS = [ # structured as (subject_id, trial_id)
    (1, 1), (1, 2), 
    (2, 0), (2, 4),
    (3, 0), (3, 1),
    (4, 0), (4, 1),
    (7, 0), (7, 1),
    (10, 0), (10, 1)
]
lite_files = [f'sub_{subject_id}_trial{trial_id:03}.h5.zip' for subject_id, trial_id in NEUROPROBE_LITE_SUBJECT_TRIALS]

# Create braintreebank directory if it doesn't exist
if not os.path.exists('braintreebank_zip'):
    os.makedirs('braintreebank_zip')

# Get the main page
url = 'https://braintreebank.dev/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links on the page
links = soup.find_all('a')

# Download each linked file
for link in links:
    href = link.get('href')
    if href:
        # Make absolute URL if relative
        file_url = urljoin(url, href)
        
        # Get filename from URL
        filename = os.path.basename(file_url)
        
        if filename:  # Only proceed if there's a filename
            if (filename == 'brain_treebank_code_release') or (filename == '2411.08343'):
                continue

            # Check lite mode filtering
            if args.lite:
                # If filename starts with sub_ but is not in our lite list, skip it
                if filename.startswith('sub_') and filename not in lite_files:
                    print(f'Skipping {filename} - not in lite mode list')
                    continue

            filepath = os.path.join('braintreebank_zip', filename)
            
            # Check if file exists and is complete
            skip_download = False
            if os.path.exists(filepath):
                try:
                    # Get file size from server
                    response = requests.head(file_url)
                    expected_size = int(response.headers.get('content-length', 0))
                    
                    # Get local file size
                    actual_size = os.path.getsize(filepath)
                    
                    if expected_size == actual_size:
                        print(f'Skipping {filename} - already downloaded')
                        skip_download = True
                except:
                    # If any error occurs during size check, re-download to be safe
                    pass
            
            if skip_download:
                continue

            print(f'Downloading {filename}...')
            # Download the file with streaming to handle large files
            file_response = requests.get(file_url, stream=True)
            
            # Save to braintreebank directory
            with open(filepath, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f'Downloaded {filename}')

print("=== DOWNLOADING DONE ===")

"""
Extract zip files in braintreebank_zip directory to braintreebank directory
"""

import os
import zipfile
# Create braintreebank directory if it doesn't exist
if not os.path.exists('braintreebank'):
    os.makedirs('braintreebank')

# Extract zip files
successful = 0
failed = 0
for filename in os.listdir('braintreebank_zip'):
    if filename.endswith('.zip'):
        print(f'Extracting {filename}...')
        try:
            with zipfile.ZipFile(os.path.join('braintreebank_zip', filename), 'r') as zip_ref:
                zip_ref.extractall('braintreebank')
            # Delete the zip file after successful extraction
            os.remove(os.path.join('braintreebank_zip', filename))
            print(f'Done. Deleted {filename} from braintreebank_zip directory because it has been extracted.')
            successful += 1
        except:
            print(f'Failed to extract.')
            failed += 1
    else:
        # Move non-zip files to braintreebank directory
        print(f'Moving {filename}...')
        try:
            import shutil
            source_path = os.path.join('braintreebank_zip', filename)
            dest_path = os.path.join('braintreebank', filename)
            shutil.move(source_path, dest_path)
            print(f'Done. Moved {filename} to braintreebank directory.')
            successful += 1
        except:
            print(f'Failed to move.')
            failed += 1

# Check if braintreebank_zip directory is empty after extraction
if os.path.exists('braintreebank_zip'):
    remaining_files = [f for f in os.listdir('braintreebank_zip') if not f.startswith('.')]
    if len(remaining_files) == 0:
        try:
            os.rmdir('braintreebank_zip')
            print("Removed empty braintreebank_zip directory")
        except OSError as e:
            print(f"Could not remove braintreebank_zip directory: {e}")
    else:
        print(f"braintreebank_zip directory still contains {len(remaining_files)} files, not removing")

print(f'\nExtraction complete: {successful} files extracted successfully, {failed} files failed')