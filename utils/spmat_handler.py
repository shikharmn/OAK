import numpy as np
import scipy.sparse as sp
import xclib.data.data_utils as du
from tqdm import tqdm, trange

def obtain_topk(preds_, k, threshold=0):
    coo_rows = np.repeat(np.arange(preds_.shape[0]), k).astype(np.int64)
    coo_cols = np.zeros(coo_rows.shape, dtype=np.int64)
    coo_data = np.zeros(coo_rows.shape, dtype=np.float32)
    
    for idx in trange(preds_.shape[0]):
        data = preds_[idx].data
        indices = preds_[idx].indices
        if len(data) == 0:
            data = np.zeros(k, dtype=np.float32)
            indices = np.zeros(k, dtype=np.int64)
        elif len(data) < k:
            data = np.concatenate([data, np.zeros(k - len(data), dtype=np.float32)])
            indices = np.concatenate([indices, np.zeros(k - len(indices), dtype=np.int64)])
        topk = np.argsort(data)[::-1][:k]
        coo_cols[idx * k: (idx + 1) * k] = indices[topk]
        coo_data[idx * k: (idx + 1) * k] = data[topk]
    
    coo_data[coo_data < threshold] = 0
    
    topk_preds = sp.coo_matrix(
        (coo_data, (coo_rows, coo_cols)),
        shape=(preds_.shape[0], preds_.shape[1])
    ).tocsr()
    
    topk_preds.eliminate_zeros()
    
    return topk_preds

def _load_spmat(path):
    extension = path[-3:]
    if extension == 'txt':
        return du.read_sparse_file(path)
    elif extension == 'npz':
        return sp.load_npz(path)
    else:
        return NotImplementedError

def prepare_metadata_spmat(cfg):
    if cfg.ngame:
        return []
    tst_retr_path = f"{cfg.data.data_dir}/{cfg.data.retrieved_eval}"
    trn_gthr_path = f"{cfg.data.data_dir}/{cfg.data.gthread_train}"
    tst_gthr_path = f"{cfg.data.data_dir}/{cfg.data.gthread_eval}"
    
    train_docmeta = _load_spmat(trn_gthr_path)
    test_docmeta_golden = _load_spmat(tst_gthr_path)        
    if cfg.retrieved_eval:
        test_docmeta_retrieved = obtain_topk(_load_spmat(tst_retr_path),
                                             cfg.data.meta_test_topk, -5)
        return train_docmeta, test_docmeta_golden, test_docmeta_retrieved
    else:
        return train_docmeta, test_docmeta_golden
    
    
def prepare_sparse_mats(cfg):
    trnXY_path = f"{cfg.data.data_dir}/{cfg.data.trn_file}"
    tstXY_path = f"{cfg.data.data_dir}/{cfg.data.tst_file}"
    trn_X_Y = _load_spmat(trnXY_path)
    tst_X_Y = _load_spmat(tstXY_path) if cfg.data.eval_interval != -1 else sp.csr_matrix((1,1))
    
    if cfg.data.right_prefix != "label":
        trn_X_Y = sp.hstack([sp.eye(trn_X_Y.shape[0]), trn_X_Y]).tocsr()
        tst_X_Y = sp.hstack([sp.eye(tst_X_Y.shape[0]), tst_X_Y]).tocsr()
    
    print(f"Mats' shape - trn: {trn_X_Y.shape}, tst: {tst_X_Y.shape}")
    
    return trn_X_Y, tst_X_Y