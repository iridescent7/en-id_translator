import requests
import zipfile
import tarfile
import spacy
import os

from pathlib import Path
from tqdm import tqdm
from munch import Munch
from collections import Counter

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

    dest = path / name

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
    def _parallel_dataset(path: Path, prefix: str='parallel'):
        url = 'https://codeload.github.com/prasastoadi/parallel-corpora-en-id/zip/refs/heads/master'
        dest = download_file(url, path)

        with zipfile.ZipFile(dest, 'r') as zf:
            for f in zf.namelist():
                if f.endswith('.tgz'):

                    with zf.open(f, 'r') as fo, tarfile.open(fileobj=fo, mode='r') as tf:
                        for m in tf.getmembers():
                            if m.name.endswith('.en') or '-EN-' in m.name:
                                tf.extract(m, path=path / prefix /'en')
                            else:
                                tf.extract(m, path=path / prefix / 'id')

        dest.unlink()
        
    @staticmethod
    def _bilingual_dataset(path: Path, prefix: str='bilingual'):
        url = 'https://github.com/desmond86/Indonesian-English-Bilingual-Corpus/archive/refs/heads/master.zip'
        dest = download_file(url, path)

        with zipfile.ZipFile(dest, 'r') as zf:
            for f in zf.namelist():
                en = f.endswith('.en')
                id = f.endswith('.id')

                if en or id:
                    dest_path = path / prefix / ('en' if en else 'id')
                    dest_path.mkdir(parents=True, exist_ok=True)

                    with zf.open(f, 'r') as fo, open(dest_path / os.path.basename(f), 'wb') as t:
                        t.write(fo.read())

        dest.unlink()
        
    @staticmethod
    def _talpco_dataset(path: Path, prefix: str='talpco'):
        url = 'https://github.com/matbahasa/TALPCo/archive/refs/heads/master.zip'
        dest = download_file(url, path)

        path_en = path / prefix / 'en'
        path_id = path / prefix / 'id'

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
                        
                        with zf.open(f, 'r') as fo, open(p / fn, 'w') as t:
                            items = []
                            for line in fo.readlines():
                                split = line.decode('UTF-8').split('\t')
                
                                if len(split) > 1:
                                    items.append(split[1].replace('\r', ''))
                            
                            t.writelines(items)
                            
        dest.unlink()
        
    @staticmethod
    def download_all(path: str='data'):
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        Dataset._parallel_dataset(path)
        Dataset._bilingual_dataset(path)
        Dataset._talpco_dataset(path)

    @staticmethod
    def load_dataset(path: str='data', langs: list=['en', 'id'], init_vocab: list[str]=[]):
        path = Path(path)

        if not path.exists():
            raise RuntimeError('No data is found, please download dataset first')

        dataset = Munch(data=dict(), langs=langs)

        for lang in langs:
            sentences = []
            for name in path.iterdir():
                for d in (name / lang).iterdir():
                    if d.is_file():
                        with open(d, 'r', encoding='UTF-8') as file:
                            # All sentences are already separated by a newline
                            for line in file.readlines():
                                sentences.append(line.lower().rstrip('\n'))

            # try to load spacy pretrained language model if supported
            try:
                tokenizer = spacy.load(lang)
            except:
                tokenizer = spacy.blank(lang)

            vocab_counter = Counter(init_vocab)
            sent_words = []

            for sent in tqdm(sentences, f'Tokenizing data ({lang})'):
                words = tokenizer(sent)

                sent_words.append(words)
                vocab_counter.update(words)

            vocab = dict(zip(vocab_counter.keys(), range(len(vocab_counter))))

            # sent_tokens = []
            # for sw in sent_words:
            #     sent_tokens.append([vocab[w] if w in vocab.keys() else vocab['<unk>'] for w in sw])

            dataset.data[lang] = Munch(vocab=vocab, sent=sentences, sent_tokens=sent_words, tokenizer=tokenizer)

        return dataset