import re
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, path: Path):
    resp = requests.get(url, stream=True)

    content_disp = resp.headers['content-disposition']

    if content_disp:
        pattern = 'filename='
        index = content_disp.find(pattern)

        if index > -1:
            name = content_disp[index+len(pattern):]
        else:
            name = url.split('/')[-1]
    else:
        name = url.split('/')[-1]

    dest = path.joinpath(name)

    tqdm_args = {
        'desc': name,
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
    def _parallel_dataset(path: Path):
        url = 'https://codeload.github.com/prasastoadi/parallel-corpora-en-id/zip/refs/heads/master'
        dest = download_file(url, path)

        with zipfile.ZipFile(dest, 'r') as zf:
            for f in zf.namelist():
                if f.endswith('.tgz'):

                    with zf.open(f, 'r') as fo, tarfile.open(fileobj=fo, mode='r') as tf:
                        for m in tf.getmembers():
                            if m.name.endswith('.en') or '-EN-' in m.name:
                                tf.extract(m, path=path.joinpath('parallel-dataset/en'))
                            else:
                                tf.extract(m, path=path.joinpath('parallel-dataset/id'))

        dest.unlink()
        
    @staticmethod
    def _bilingual_dataset(path: Path):
        url = 'https://github.com/desmond86/Indonesian-English-Bilingual-Corpus/archive/refs/heads/master.zip'
        dest = download_file(url, path)

        with zipfile.ZipFile(dest, 'r') as zf:
            for f in zf.namelist():
                if f.endswith('.en'):
                    zf.extract(f, path.joinpath('bilingual/en'))
                elif f.endswith('.id'):
                    zf.extract(f, path.joinpath('bilingual/id'))

        dest.unlink()
        
    @staticmethod
    def _talpco_dataset(path: Path):
        url = 'https://github.com/matbahasa/TALPCo/archive/refs/heads/master.zip'
        dest = download_file(url, path)

        path_en = path.joinpath('talpco/en')
        path_id = path.joinpath('talpco/id')

        path_en.mkdir(parents=True, exist_ok=True)
        path_id.mkdir(exist_ok=True)

        file_to_path = [
            ('data_eng.txt', path_en),
            ('data_ind.txt', path_id)
        ]

        with zipfile.ZipFile(dest, 'r') as zf:
            for f in zf.namelist():
                for fn, p in file_to_path:
                    if f.endswith(fn):
                        
                        with zf.open(f, 'r') as fo, open(p.joinpath(fn), 'w') as t:
                            for l in fo.readlines():
                                item = l.split(b'\t')

                                if len(item) > 1:
                                    t.write(item[1])
                            
        dest.unlink()
        
    @staticmethod
    def download_all(path='data'):
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        Dataset._parallel_dataset(path)
        Dataset._bilingual_dataset(path)
        Dataset._talpco_dataset(path)


