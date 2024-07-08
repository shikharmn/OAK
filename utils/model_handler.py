import os
import torch
import numpy as np
from tqdm import trange

normalize = lambda x: (x.T / np.linalg.norm(x, axis=1)).T

class MyDataParallel(torch.nn.DataParallel):
    """Allows data parallel to work with methods other than forward"""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def get_metadata_embs(cfg, encoder):
    emb_dump_path = f"{cfg.base_dir}/{cfg.data.pretrained_weight_path}"
    os.makedirs("/".join(emb_dump_path.split('/')[:-1]), exist_ok=True)
    dim = 768 if cfg.data.transform_dim == -1 else cfg.data.transform_dim
    encoder.eval().to(cfg.device)
    input_ids = np.memmap(
        f"{cfg.data.tokenization_folder}/{cfg.data.meta_prefix}_input_ids.dat",
        mode='r', dtype=cfg.data.tok_dtype).reshape(-1, cfg.data.max_length_left)
    attention_mask = np.memmap(
        f"{cfg.data.tokenization_folder}/{cfg.data.meta_prefix}_attention_mask.dat",
        mode='r', dtype=cfg.data.tok_dtype).reshape(-1, cfg.data.max_length_left)
    embeddings = np.zeros((input_ids.shape[0], dim))
    print("Updating metadata representations.")
    with torch.no_grad():
        for i in trange(0, input_ids.shape[0], cfg.data.eval_bsz):
            batch_input_ids = torch.LongTensor(input_ids[i: i + cfg.data.eval_bsz]).to(cfg.device)
            batch_attention_mask = torch.LongTensor(attention_mask[i: i + cfg.data.eval_bsz]).to(cfg.device)
            _batch_embeddings = encoder(
                batch_input_ids, batch_attention_mask).cpu().numpy()
            embeddings[i: i + batch_input_ids.shape[0]] = _batch_embeddings    
    
    np.save(emb_dump_path, embeddings)
    return embeddings

def load_emb_weights(cfg, snet, force_overwrite=False):
    emb_path = f"{cfg.base_dir}/{cfg.data.pretrained_weight_path}"
    if os.path.exists(emb_path) and not force_overwrite:
        weights_np = np.load(emb_path)
    else:
        weights_np = get_metadata_embs(cfg, snet.encoder)
    
    weights = snet.state_dict()
    print("Returning modified state dict.")
    weights['embs.weight'] = torch.from_numpy(weights_np)
    return weights