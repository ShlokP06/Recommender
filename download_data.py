import gdown
import os

def download_files():
    dic = {"credits.csv":"1wcJ7pGVfreq7xzhSNlqyn11g21axDXOl",
           "keywords.csv":"1AtlqGv01-VyqETCh3LtDffRCP8n5DR2E",
           "movies_metadata.csv":"1zG0sTTK86z-apfBjPe3rPcUyziNNAXOE",
           "links.csv":"1AWcNcoyJ5ES2qBBCNEwKsBcZeUFNgZj1",
           "links_small.csv":"1FVLE7vszAYnODgcCAZBJN1wGQCujyk7s",
           "ratings_small.csv":"1BQWLWK0pFb3oN7zKrdb5YqxJVXm8DydY"}

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    for key in dic:
        value = dic[key]
        output_path = os.path.join(output_dir,key)
        if not os.path.exists(output_path):
            print(f"Downloading {key} with gdown...")
            gdown.download(id=value, output=output_path, quiet=False)
            print("Download Complete")
        else:
            print(f"{key} Already Exists..")

if __name__ == '__main__':
    download_files()