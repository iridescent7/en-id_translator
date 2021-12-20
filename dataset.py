import os
import re
import io
import requests
import zipfile
import tarfile 
from tqdm import tqdm

def download_file(url, path):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    
    if 'content-disposition' in resp.headers.keys():
        name = re.search('filename=(.+)', resp.headers['content-disposition']).group()
    else:
        name = url.split('/')[-1]

    dest = os.path.join(path, name)

    tqdm_args = {
        'desc': path,
        'total': int(resp.headers.get('content-length', 0)),
        'unit': 'iB',
        'unit_scale': True,
        'unit_divisor': 1024
    }

    with open(dest, 'wb') as file, tqdm(**tqdm_args) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    return dest

class Dataset():
    @staticmethod
    def _parallel_dataset(path):
        url = 'https://codeload.github.com/prasastoadi/parallel-corpora-en-id/zip/refs/heads/master'
        dest = download_file(url, path)

        with zipfile.ZipFile(dest) as zf:
            for f in zf.namelist():
                if f.endswith('.tgz'):
                    content = io.BytesIO(zf.read(f))

                    with tarfile.open(fileobj=content) as tf:
                        for m in tf.getmembers():
                            if m.name.endswith('.en') or '-EN-' in m.name:
                                tf.extract(m, path=os.path.join(path, 'parallel-dataset/en'))
                            else:
                                tf.extract(m, path=os.path.join(path, 'parallel-dataset/id'))

        os.remove(dest)
        
    @staticmethod
    def download_all(path='data'):
        if not os.path.isdir(path):
            os.mkdir(path)
        
        Dataset._parallel_dataset(path)


