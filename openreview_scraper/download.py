import pandas as pd
import time
import json
import ipdb
st = ipdb.set_trace

def read_papers(fpath='attention.csv'):
    """Read papers from CSV file"""
    try:
        papers = pd.read_csv(fpath)
        return papers
    except FileNotFoundError:
        print(f"File {fpath} not found")
        return None


import requests
import os
from urllib.parse import urlparse

def download_pdf(url, output_dir='pdfs', filename=None):
    """Download PDF from URL and save to output directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # st()
    try:
        # Get filename from URL
        # filename = os.path.basename(urlparse(url).path)
        # if not filename.endswith('.pdf'):
        #     filename += '.pdf'
            
        output_path = os.path.join(output_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(output_path):
            print(f"File {filename} already exists, skipping...")
            return False
            
        # Download PDF
        response = requests.get(url)
        response.raise_for_status()
        
        # Save PDF
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        print(f"Downloaded {filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except Exception as e:
        print(f"Error saving {url}: {e}")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Download PDFs from OpenReview CSV')
    parser.add_argument('--csv', type=str, default='attention',
                      help='Path to CSV file containing paper info (default: attention.csv)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    root_dir = f"/grogu/user/mprabhud/papers_o1"
    papers = read_papers(fpath=f"{args.csv}.csv")
    output_dir = f"{root_dir}/{args.csv}_papers"
    os.makedirs(output_dir, exist_ok=True)
    for idx, pdf in enumerate(papers.pdf):
        print(f"Processing paper {idx}")
        pdf_dir = f"{output_dir}/{idx:05d}" 
        additional_info = papers.iloc[idx].to_dict()
        os.makedirs(pdf_dir, exist_ok=True)
        indexes = [i for i in range(len(pdf)) if pdf.startswith("http", i)]
        pdf_new = pdf[indexes[-1]:]
        
        # if "openreview" in pdf_new and "https://www.ecva.net" in pdf_new:
        #     pdf_new = pdf_new.replace("https://openreview.net", "")
        
        # if "openreview" in pdf_new and "arxiv" in pdf_new:
        #     pdf_new = pdf_new.replace("https://openreview.net", "")
        
        success = download_pdf(pdf_new, filename=f"main.pdf", output_dir=pdf_dir)

        # Save additional paper info as JSON
        info_path = os.path.join(pdf_dir, 'info.json')
        # st()
        if not os.path.exists(info_path):
            with open(info_path, 'w') as f:
                json.dump(additional_info, f, indent=2)
        if success:
            time.sleep(5)
    print("done")