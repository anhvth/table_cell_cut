import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--link', required=True, help='Format: https://drive.google.com/open?id=fid')
parser.add_argument('--to', required=True, help='Format link to file')
args = parser.parse_args()
fid = args.link.split('=')[-1]
import requests

def download_file_from_google_drive(id, destination):
    print('DOWNLOADING: ', id)
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    file_id = fid
    destination = args.to
    download_file_from_google_drive(file_id, destination)