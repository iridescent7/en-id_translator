import requests
import zipfile
import tarfile
import pickle
from pathlib import Path
from tqdm import tqdm
from munch import Munch
from nltk import sent_tokenize, word_tokenize
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
    def _parallel_dataset(path: Path):
        url = 'https://codeload.github.com/prasastoadi/parallel-corpora-en-id/zip/refs/heads/master'
        dest = download_file(url, path)

        with zipfile.ZipFile(dest, 'r') as zf:
            for f in zf.namelist():
                if f.endswith('.tgz'):

                    with zf.open(f, 'r') as fo, tarfile.open(fileobj=fo, mode='r') as tf:
                        for m in tf.getmembers():
                            if m.name.endswith('.en') or '-EN-' in m.name:
                                tf.extract(m, path=path / 'parallel-dataset/en')
                            else:
                                tf.extract(m, path=path / 'parallel-dataset/id')

        dest.unlink()
        
    @staticmethod
    def _bilingual_dataset(path: Path):
        url = 'https://github.com/desmond86/Indonesian-English-Bilingual-Corpus/archive/refs/heads/master.zip'
        dest = download_file(url, path)

        with zipfile.ZipFile(dest, 'r') as zf:
            for f in zf.namelist():
                if f.endswith('.en'):
                    zf.extract(f, path / 'bilingual/en')
                elif f.endswith('.id'):
                    zf.extract(f, path / 'bilingual/id')

        dest.unlink()
        
    @staticmethod
    def _talpco_dataset(path: Path):
        url = 'https://github.com/matbahasa/TALPCo/archive/refs/heads/master.zip'
        dest = download_file(url, path)

        path_en = path / 'talpco/en'
        path_id = path / 'talpco/id'

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
                                    items.append(split[1])
                            
                            t.write('\n'.join(items))
                            
        dest.unlink()
        
    @staticmethod
    def download_all(path: str='raw_data'):
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        Dataset._parallel_dataset(path)
        Dataset._bilingual_dataset(path)
        Dataset._talpco_dataset(path)

    @staticmethod
    def build_dataset(path: str='raw_data', new_path: str='data', langs: list=['en', 'id']):
        path = Path(path)

        if not path.exists():
            raise RuntimeError('No data is found, please download dataset first')

        new_path = Path(new_path)
        new_path.mkdir(exist_ok=True)

        dataset = dict()

        for lang in langs:
            dataset[lang] = []

            data_path = new_path / (lang + '.txt')
            vocab_path = new_path / (lang + '_vocab.txt')
            encoded_path = new_path / (lang + '_encoded.bin')

            print(f'[{lang}] Reading all data into memory...')

            for name in path.iterdir():
                with open(data_path, 'w', encoding='UTF-8') as df, \
                     open(vocab_path, 'w', encoding='UTF-8') as vf:

                    lang_path = name / lang
                    dirs = [dir for dir in lang_path.iterdir()]

                    for dir in dirs:
                        if dir.is_file():
                            with open(dir, 'r', encoding='UTF-8') as f:
                                dataset[lang].append(f.read())

            print(f'[{lang}] Tokenizing data...')
            vocab_counter = Counter(['<bos>', '<eos>'])
            
            sents = sent_tokenize('\n'.join(dataset[lang]))
            sent_words = []

            for sent in sents:
                words = word_tokenize(sent)

                sent_words.append(words)
                vocab_counter.update(words)
            
            print(f'[{lang}] Building input features...')
            vocab_count = len(vocab_counter)

            vocab_idx = vocab_counter
            vocab_encoded = dict()

            for i, k in enumerate(vocab_idx.keys()):
                onehot = [0] * vocab_count
                onehot[i] = 1

                vocab_idx[k] = i
                vocab_encoded[k] = onehot

            BOS_IDX = vocab_encoded['<bos>']
            EOS_IDX = vocab_encoded['<eos>']

            features = []
            for sw in sent_words:
                features.append([vocab_encoded[BOS_IDX]] + [vocab_encoded[w] for w in sw] + [vocab_encoded[EOS_IDX]])

            print(f'[{lang}] Saving to disk...')

            with open(encoded_path, 'wb') as f:
                pickle.dump(f, features)

            df.write('\n'.join(sents))
            vf.write('\n'.join(vocab_counter.keys()))