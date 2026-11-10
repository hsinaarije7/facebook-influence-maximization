import urllib.request
import gzip
import shutil
import os

# Download dataset
url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
gz_file = "facebook_combined.txt.gz"
txt_file = "facebook_combined.txt"

print("Downloading dataset...")
urllib.request.urlretrieve(url, gz_file)

print("Extracting dataset...")
with gzip.open(gz_file, "rb") as f_in:
    with open(txt_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Done!")
print("Saved as:", os.path.abspath(txt_file))