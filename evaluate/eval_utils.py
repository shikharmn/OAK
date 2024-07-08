import os
import gc
import time
import torch
import functools
import numpy as np
import scipy.sparse as sp
from tqdm import trange, tqdm
from xclib.utils.matrix import SMatrix
from xclib.evaluation import xc_metrics
from xclib.data import data_utils as du
from sklearn.preprocessing import normalize
from xclib.utils.sparse import csr_from_arrays
import xclib.evaluation.xc_metrics as xc_metrics
# from xclib.utils.shortlist import Shortlist


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


from contextlib import contextmanager
@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()

def self_attention(seq_len, dtype):
    mask = np.ones((12, seq_len, seq_len), dtype=dtype) * float('-inf')
    mask[:, :seq_len, :seq_len] = 0
    return mask

def cross_attention(seq_len, dtype):
    mask = np.ones((12, seq_len, seq_len), dtype=dtype) * float('-inf')
    mask[:, np.arange(seq_len), np.arange(seq_len)] = 0
    mask[:, :, 0] = 0
    return mask

@timeit
def get_prf_embeddings(tokenization_folder, dump_folder, prefix, model, max_len, dim, dtype='int64', bsz=2000, algorithm="prf"):
    """Get embeddings for given tokenized files"""
    input_ids = np.memmap(
        f"{tokenization_folder}/{prefix}_prf_input_ids.dat",
        mode='r', dtype=dtype).reshape(-1, max_len)
    attention_mask = np.memmap(
        f"{tokenization_folder}/{prefix}_prf_attention_mask.dat",
        mode='r', dtype=dtype).reshape(-1, max_len)
    embeddings = np.memmap(
        f"{dump_folder}/{prefix}.{algorithm}.dat",
        mode='w+', dtype=np.float32, shape=(input_ids.shape[0], dim))
    with evaluating(model), torch.no_grad():
        for i in trange(0, input_ids.shape[0], bsz):
            batch_input_ids = torch.LongTensor(input_ids[i: i + bsz])
            batch_attention_mask = torch.LongTensor(attention_mask[i: i + bsz])
            _batch_embeddings = model(
                batch_input_ids, batch_attention_mask, None, None).cpu().numpy()
            embeddings[i: i + batch_input_ids.shape[0]] = _batch_embeddings

    return embeddings

@timeit
def get_bert_embeddings(tokenization_folder, dump_folder, prefix, model, max_len, dim, dtype='int64', bsz=2000, algorithm="ngame"):
    """Get embeddings for given tokenized files"""
    input_ids = np.memmap(
        f"{tokenization_folder}/{prefix}_input_ids.dat",
        mode='r', dtype=dtype).reshape(-1, max_len)
    attention_mask = np.memmap(
        f"{tokenization_folder}/{prefix}_attention_mask.dat",
        mode='r', dtype=dtype).reshape(-1, max_len)
    embeddings = np.memmap(
        f"{dump_folder}/{prefix}.{algorithm}.dat",
        mode='w+', dtype=np.float32, shape=(input_ids.shape[0], dim))
    with evaluating(model), torch.no_grad():
        for i in trange(0, input_ids.shape[0], bsz):
            batch_input_ids = torch.LongTensor(input_ids[i: i + bsz])
            batch_attention_mask = torch.LongTensor(attention_mask[i: i + bsz])
            _batch_embeddings = model(
                None, None, batch_input_ids, batch_attention_mask).cpu().numpy()
            embeddings[i: i + batch_input_ids.shape[0]] = _batch_embeddings

    return embeddings

@timeit
def get_embeddings_comb(tokenization_folder, dump_folder, meta_prefix, prefix, model, max_len, dim, max_titles, dtype='int64', bsz=2000, algorithm="ngame", retriever_smat=None):
    """Get embeddings for given tokenized files"""
    device = 'cuda:0'
    input_ids = np.memmap(
        f"{tokenization_folder}/{prefix}_input_ids.dat",
        mode='r', dtype=dtype).reshape(-1, max_len)
    attention_mask = np.memmap(
        f"{tokenization_folder}/{prefix}_attention_mask.dat",
        mode='r', dtype=dtype).reshape(-1, max_len)
    if retriever_smat is None:
        doc_meta = du.read_sparse_file(f"{tokenization_folder}/../{meta_prefix}_tst_X_Y.txt")
    else:
        doc_meta = retriever_smat
    meta_degree = doc_meta.getnnz(axis=1)
    
    embeddings = np.memmap(
        f"{dump_folder}/{prefix}.{algorithm}.dat",
        mode='w+', dtype=np.float32, shape=(input_ids.shape[0], dim))
    with evaluating(model), torch.no_grad():
        for i in trange(0, input_ids.shape[0], bsz):
            batch_input_ids = torch.LongTensor(input_ids[i: i + bsz])
            batch_attention_mask = torch.LongTensor(attention_mask[i: i + bsz])
            batch_title_ids = torch.from_numpy(np.zeros((batch_input_ids.shape[0], max_titles), dtype=dtype))
            batch_mean_mask = torch.from_numpy(np.zeros((batch_input_ids.shape[0], max_titles + 1), dtype=dtype))
            for _idx in range(batch_input_ids.shape[0]):
                attn_limit = min(max_titles, meta_degree[i + _idx])
                batch_title_ids[_idx, :attn_limit] = torch.from_numpy(np.random.choice(doc_meta[i + _idx].indices,
                                                                                       size=attn_limit,
                                                                                       replace=False))
                batch_mean_mask[_idx, :attn_limit + 1] = 1
                
            _batch_embeddings = model(
                batch_input_ids.to(device),
                batch_attention_mask.to(device),
                None, None,
                batch_title_ids.to(device),
                batch_mean_mask.to(device)).cpu().numpy()
            
            embeddings[i: i + batch_input_ids.shape[0]] = _batch_embeddings

    return embeddings


def get_filter_map(fname):
    """Load filter file as numpy array"""
    if fname is not None and fname != "":
        return np.loadtxt(fname).astype('int')
    else:
        return None


def filter_predictions(pred, mapping):
    """Filter predictions using given mapping"""
    if mapping is not None and len(mapping) > 0:
        print("Filtering labels.")
        pred[mapping[:, 0], mapping[:, 1]] = 0
        pred.eliminate_zeros()
    return pred


def evaluate(_true, _pred, _train, k, A, B, recall_only=False):
    """Evaluate function
    * use recall_only = True when dataset is large
    * k: used only for recall@k (precision etc., are computed till @5)
    """
    _true.indices = _true.indices.astype('int64')
    if not recall_only:
        inv_propen = xc_metrics.compute_inv_propesity(_train, A, B)
        acc = xc_metrics.Metrics(_true, inv_propen)
        acc = acc.eval(_pred, 5)
    else:
        print("Only R@k is computed. Don't be surprised with 0 val of others")
        acc = np.zeros((4, 5))
    rec = xc_metrics.recall(_pred, _true, k)  # get the recall
    return acc, rec


def evaluate_with_filter(true_labels, predicted_labels,
                         train_labels, filter_labels, k,
                         A, B, recall_only):
    """Evaluate function with support of filter file
    * use recall_only = True when dataset is large
    * k: used only for recall@k (precision etc., are computed till @5)
    """
    mapping = get_filter_map(filter_labels)
    predicted_labels = filter_predictions(predicted_labels, mapping)
    return evaluate(
        true_labels, predicted_labels, train_labels, k, A, B, recall_only)

def _predict_ova(X, clf, k=20, batch_size=32, device="cuda", return_sparse=True):
    """Predictions in brute-force manner"""
    with torch.no_grad():
        num_instances, num_labels = len(X), len(clf)
        batches = np.array_split(range(num_instances), num_instances//batch_size)
        output = SMatrix(
            n_rows=num_instances,
            n_cols=num_labels,
            nnz=k)
        X = torch.from_numpy(X)        
        clf = torch.from_numpy(clf).to(device).T   
        for ind in tqdm(batches):
            s_ind, e_ind = ind[0], ind[-1] + 1
            _X = X[s_ind: e_ind].to(device)
            ans = _X @ clf
            vals, ind = torch.topk(
                ans, k=k, dim=-1, sorted=True)
            output.update_block(
                s_ind, ind.cpu().numpy(), vals.cpu().numpy())
            del _X
        if return_sparse:
            return output.data()
        else:
            return output.data('dense')[0]


def predict_and_eval(features, clf, labels,
                     trn_labels, filter_labels,
                     A, B, k=10, mode='ova', huge=False,
                     device='cuda'):
    """
    Predict on validation set and evaluate
    * support for filter file (pass "" or empty file otherwise)
    * ova will get top-k predictions but anns would get 300 (change if required)"""
    mode='anns' if huge else mode
    if mode == 'ova':
        pred = _predict_ova(normalize(features, copy=True), normalize(clf, copy=True), k=k, batch_size=128, device=device)
    else:
        raise NotImplementedError
    labels.indices = labels.indices.astype('int64')
    print(labels.shape, pred.shape)
    acc, r = evaluate_with_filter(labels, pred, trn_labels, filter_labels, k, A, B, huge)
    return acc, r, pred

def validate(cfg, snet, trn_X_Y, val_X_Y, meta_tst_X_Y, mode="ova"):
    print("Extracting test document embeddings.")
    if cfg.ngame:
        val_doc_embeddings = get_bert_embeddings(
            cfg.data.tokenization_folder,
            f"tmp/{cfg.version}/embeddings",
            "test", snet,
            cfg.data.max_length_right,
            dtype=cfg.data.tok_dtype,
            dim=768 if cfg.data.transform_dim == -1 else cfg.data.transform_dim,
            bsz=cfg.data.eval_bsz,
            algorithm='ngame'
        )
    else:
        val_doc_embeddings = get_embeddings_comb(
            cfg.data.tokenization_folder,
            f"tmp/{cfg.version}/embeddings",
            cfg.data.meta_prefix, "test", snet,
            cfg.data.max_length_right,
            dtype=cfg.data.tok_dtype,
            dim=768 if cfg.data.transform_dim == -1 else cfg.data.transform_dim,
            max_titles=cfg.data.max_titles,
            bsz=cfg.data.eval_bsz,
            algorithm='oak',
            retriever_smat=meta_tst_X_Y[0]
        )
        if len(meta_tst_X_Y) == 2:
            val_doc_embs_retrieved = get_embeddings_comb(
                cfg.data.tokenization_folder,
                f"tmp/{cfg.version}/embeddings",
                cfg.data.meta_prefix, "test", snet,
                cfg.data.max_length_right,
                dtype=cfg.data.tok_dtype,
                dim=768 if cfg.data.transform_dim == -1 else cfg.data.transform_dim,
                max_titles=cfg.data.max_titles,
                bsz=cfg.data.eval_bsz,
                algorithm='oakretrieved',
                retriever_smat=meta_tst_X_Y[1]
            )
        else:
            val_doc_embs_retrieved = None
                
    print(f"Extracting {cfg.data.right_prefix} embeddings.")
    label_embeddings = get_bert_embeddings(
        cfg.data.tokenization_folder,
        f"tmp/{cfg.version}/embeddings",
        cfg.data.right_prefix, snet,
        cfg.data.max_length_right,
        dtype=cfg.data.tok_dtype,
        dim=768 if cfg.data.transform_dim == -1 else cfg.data.transform_dim,
        bsz=cfg.data.eval_bsz,
        algorithm='ngame' if cfg.ngame else 'oak'
        )
    
    if cfg.data.filter_labels == "":
        filter_labels = None
    else:
        filter_labels = os.path.join(cfg.data.data_dir, cfg.data.filter_labels)
        
    
    acc, r, preds = predict_and_eval(val_doc_embeddings, label_embeddings, val_X_Y, trn_X_Y,
                                     filter_labels, A=cfg.data.A, B=cfg.data.B, k=100, mode=mode, huge=False)
    sp.save_npz(f"{cfg.data.model_dir}/preds_mat.npz", preds)
    acc, r = [acc], [r]
    if len(meta_tst_X_Y) == 2:
        acc_ret, r_ret, preds = predict_and_eval(val_doc_embs_retrieved, label_embeddings, val_X_Y, trn_X_Y,
                                                 filter_labels, A=cfg.data.A, B=cfg.data.B, k=100, mode=mode, huge=False)
        sp.save_npz(f"{cfg.data.model_dir}/preds_mat_retrieved.npz", preds)
        acc += [acc_ret]
        r += [r_ret]
        del val_doc_embs_retrieved
    del val_doc_embeddings, label_embeddings
    gc.collect()
    return acc, r