import requests
import zipfile
import tarfile
import os
import spacy
import numpy as np

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
    def download_all(path: str='raw_data'):
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        print('Downloading data')

        Dataset._parallel_dataset(path)
        Dataset._bilingual_dataset(path)
        Dataset._talpco_dataset(path)

        print('Finished downloading')

    @staticmethod
    def build_dataset(path: str='data',
                      data_path: str='raw_data',
                      langs: list[str]=['en', 'id'],
                      test_size: float=0.2,
                      random_state: int=42):
        path = Path(path)        
        data_path = Path(data_path)

        if not data_path.exists():
            raise RuntimeError('No data is found, please download dataset first')

        path.mkdir(exist_ok=True)

        data = dict()
        last_len = -1

        print('Building dataset')

        for lang in langs:
            sent = []
            
            for name in data_path.iterdir():
                for d in (name / lang).iterdir():
                    if d.is_file():
                        with open(d, 'r', encoding='UTF-8') as file:
                            # All sentences are already separated by a newline
                            for line in file.readlines():
                                sent.append(line.lower())

            data[lang] = np.array(sent)

            if last_len == -1:
                last_len = len(sent)
            elif last_len != len(sent):
                raise RuntimeError('Data length of sentence mismatch ({} -> {} ({}))'.format(last_len, len(sent), lang))


        state = np.random.get_state()
        np.random.seed(random_state)

        msk = np.random.rand(last_len) > test_size
        np.random.set_state(state)
        
        for lang in langs:
            with open(path / (lang + '_train.txt'), 'w', encoding='UTF-8') as file:
                file.writelines(data[lang][msk])
                
            with open(path / (lang + '_test.txt'), 'w', encoding='UTF-8') as file:
                file.writelines(data[lang][~msk])
    
        print('Finished building dataset')

    @staticmethod
    def load_dataset(path: str='data',
                      src_lang: str='en',
                      tgt_lang: str='id'):
        path = Path(path)

        if not path.exists():
            raise RuntimeError('No data is found, please download and/or build dataset first')

        data = dict()
        vocab = dict()
        tokenizer = dict()

        for lang in [src_lang, tgt_lang]:
            try:
                tokenizer[lang] = spacy.load(lang)
            except:
                tokenizer[lang] = spacy.blank(lang)

        for type in ['train', 'test']:
            data[type] = dict()

            for lang in [src_lang, tgt_lang]:
                sent = []

                with open(path / (lang + '_' + type + '.txt'), 'r', encoding='UTF-8') as file:
                    for line in file.readlines():
                        sent.append(line.lower().rstrip('\n'))

                sent_tokens = []

                for sent in tqdm(sent, f'Tokenizing data ({type}/{lang})'):
                    sent_tokens.append(tokenizer[lang](sent))

                if type == 'train':
                    vocab_counter = Counter(['<unk>', '<pad>', '<sos>', '<eos>'])
                    vocab_counter.update(sent_tokens)

                    vocab[lang] = dict(zip(vocab_counter.keys(), range(len(vocab_counter))))

                token_ids = []
                for sent_tok in sent_tokens:
                    token_ids.append([vocab[lang]['<sos>']] +
                                     [vocab[lang][tok] if tok in vocab[lang].keys() else vocab[lang]['<unk>'] for tok in sent_tok] +
                                     [vocab[lang]['<eos>']])

                data[type][lang] = Munch(vocab=vocab, token_ids=token_ids, tokenizer=tokenizer)

        return data