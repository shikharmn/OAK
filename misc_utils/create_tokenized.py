"""
For a dataset, create tokenized files in the folder {tokenizer-type}-{maxlen} folder inside the database folder
Sample usage: python -W ignore -u create_tokenized_files.py --data-dir /scratch/Workspace/data/LF-AmazonTitles-131K --tokenizer-type bert-base-uncased --max-length 32 --dump-folder-name bert-base-uncased-32
"""
import torch.multiprocessing as mp
from tqdm import trange
from transformers import AutoTokenizer
import os
import numpy as np
import time
import functools
import argparse


def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time
        print("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value
    return wrapper_timer


def _tokenize(batch_input):
    tokenizer, max_len, batch_corpus = batch_input[0], batch_input[1], batch_input[2]
    temp = tokenizer.batch_encode_plus(
                    batch_corpus,                           # Sentence to encode.
                    add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                    max_length = max_len,                   # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,           # Construct attn. masks.
                    return_tensors = 'np',                  # Return numpy tensors.
                    truncation=True
            )

    return (temp['input_ids'], temp['attention_mask'])

def convert(corpus, tokenizer, max_len, num_threads, bsz=100000): 
    batches = [(tokenizer, max_len, corpus[batch_start: batch_start + bsz]) for batch_start in range(0, len(corpus), bsz)]

    pool = mp.Pool(num_threads)
    batch_tokenized = pool.map(_tokenize, batches)
    pool.close()

    input_ids = np.vstack([x[0] for x in batch_tokenized])
    attention_mask = np.vstack([x[1] for x in batch_tokenized])

    del batch_tokenized 

    return input_ids, attention_mask

@timeit
def tokenize_dump_memmap(corpus, tokenization_dir, tokenizer, max_len, prefix, num_threads, batch_size=10000000, dtype='int64'):
    ii = np.memmap(f"{tokenization_dir}/{prefix}_input_ids.dat", dtype=dtype, mode='w+', shape=(len(corpus), max_len))
    am = np.memmap(f"{tokenization_dir}/{prefix}_attention_mask.dat", dtype=dtype, mode='w+', shape=(len(corpus), max_len))
    for i in trange(0, len(corpus), batch_size):
        _input_ids, _attention_mask = convert(corpus[i: i + batch_size], tokenizer, max_len, num_threads)
        ii[i: i + _input_ids.shape[0], :] = _input_ids
        am[i: i + _input_ids.shape[0], :] = _attention_mask

parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", type=str, required=True, help="Data directory path - with {trn,tst}_X.txt, {trn,tst}_X_Y.txt and Y.txt")
parser.add_argument("--max-length", type=int, help="Max length for tokenizer", default=32)
parser.add_argument("--tokenizer-type", type=str, help="Tokenizer to use", default="bert-base-uncased")
parser.add_argument("--num-threads", type=int, help="Number of threads to use", default=24)
parser.add_argument("--dump-folder-name", type=str, help="Dump folder inside dataset folder", default="")
parser.add_argument("--dtype", default='int64')


args = parser.parse_args()

text_corpus_prefixes = ["label", "train", "test", "related_items"]

max_len = args.max_length
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type, do_lower_case=True)
if(args.dump_folder_name == ""):
    dump_folder_name = f"{args.tokenizer_type}-{max_len}"
else:
    dump_folder_name = args.dump_folder_name
tokenization_dir = f"{args.data_dir}/{dump_folder_name}"
os.makedirs(tokenization_dir, exist_ok=True)

print(f"Dumping files in {tokenization_dir}...")

for prefix in text_corpus_prefixes:
    text = [x.strip() for x in open(f'{args.data_dir}/raw_data/{prefix}.raw.txt', "r", encoding="latin").readlines()]

    print(f"Dumping for {prefix}...")
    tokenize_dump_memmap(text, tokenization_dir, tokenizer, max_len, prefix, args.num_threads, dtype=args.dtype)
