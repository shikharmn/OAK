import os
import sys
import torch
import numpy as np
from functools import partial
from transformers import AutoTokenizer, BertTokenizerFast
from data.train_dataset_x_ads import (collate_fn_ads,
                                      DatasetDocumentWise,
                                      DDocOneBatch)
from data.collate import (NGAMECollateClass,
                          OAKCollateClass)
from data.train_dataset_public import (NGAMEPublicDataset,
                                       OAKPublicDataset)
from data.collate_raw import NGAMERawCollateClass
from data.train_dataset_raw import NGAMERawClass, NGAMEMMapRawClass
from modeling.models import (NGAMENetwork,
                             OAKNetworkReg)
from utils.model_handler import MyDataParallel, load_emb_weights
from modeling.loss import TripletMarginLossOHNM, MultimodalConfidenceLoss

class MySampler(torch.utils.data.Sampler[int]):
    def __init__(self, order):
        self.order = order.copy()

    def update_order(self, x):
        self.order[:] = x[:]

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)

def prepare_loss(cfg):
    """
    Set-up the loss function
    * num_violators can be printed, if required
    * apply_softmax is more agressive (focus more on top violators)
    """
    criterionDL = TripletMarginLossOHNM(margin=cfg.data.margin, k=cfg.data.num_negatives,
                                        num_violators=cfg.data.num_violators, apply_softmax=cfg.data.agressive_loss)
    criterionDL_enc = TripletMarginLossOHNM(margin=cfg.data.margin, k=cfg.data.reg_num_negatives,
                                            num_violators=cfg.data.num_violators, apply_softmax=cfg.data.agressive_loss)
    criterionMML = MultimodalConfidenceLoss(margin=0.05, k=cfg.data.mml_num_negatives)

    return criterionDL, criterionDL_enc, criterionMML


def prepare_network(cfg):
    """
    Set-up the network

    * Use DP if multiple GPUs are available
    """
    print("==> Creating model, optimizer...")

    if cfg.ngame:
        snet = NGAMENetwork(
            cfg.data.encoder_name,
            cfg.data.encoder_type,
            cfg.data.transform_dim,
            torch.device(cfg.device),
            vocab_size=cfg.data.vocab_size,)
    else:
        snet = OAKNetworkReg(cfg.data.encoder_name,
                             cfg.data.encoder_type,
                             cfg.data.transform_dim,
                             cfg.data.n_titles,
                             torch.device(cfg.device),
                             vocab_size=cfg.data.vocab_size,
                             sparse=True)
    if not cfg.ngame:
        print(f"Loading weights from {cfg.base_dir}/{cfg.data.pretrained_weight_path}")
        state_dict = load_emb_weights(cfg, snet)
        snet.load_state_dict(state_dict)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {len(cfg.training_devices)} GPUs!")
        snet.encoder = MyDataParallel(snet.encoder, device_ids=cfg.training_devices)
    
    snet.to(torch.device(cfg.device))

    print(snet)
    return snet

def prepare_tokenizer(cfg):
    if cfg.data.encoder_type == 'st':
        return AutoTokenizer.from_pretrained(f"sentence-transformers/{cfg.data.encoder_name}", do_lower_case=True)
    elif cfg.data.vocab_file == '':
        return AutoTokenizer.from_pretrained(cfg.data.encoder_name, do_lower_case=True)
    else:
        return BertTokenizerFast(vocab_file=cfg.data.vocab_file, do_lower_case=True)

def prepare_data(cfg, trn_X_Y, meta_trn_X_Y):
    if not (os.path.exists(cfg.data.tokenization_folder)):
        if not cfg.lazy_loader:
            print("Please create tokenization memmaps for this " "dataset using CreateTokenizedFiles.py as a one time effort")
            sys.exit(0)
        else:
            os.makedirs(cfg.data.tokenization_folder, exist_ok=True)

    print("==> Creating Dataloader...")

    if cfg.ngame:
        train_dataset = NGAMEPublicDataset(trn_X_Y, 'train',
                                           cfg.data.tokenization_folder,
                                        cfg.data.max_length_left,
                                        cfg.data.max_length_right,
                                        dtype=cfg.data.tok_dtype,
                                        num_samples_test=2048 if cfg.test_mode else -1)
        collate_fn = NGAMECollateClass(cfg.data.max_length_left)
    else:
        train_dataset = OAKPublicDataset(trn_X_Y, 'train',
                                         cfg.data.tokenization_folder,
                                         cfg.data.max_length_left,
                                         cfg.data.max_length_right,
                                         cfg.data.max_titles,
                                         dtype=cfg.data.tok_dtype,
                                         doc_meta_sp=meta_trn_X_Y,
                                         num_samples_test=2048 if cfg.test_mode else -1,
                                         meta_prefix=cfg.data.meta_prefix)
        collate_fn = OAKCollateClass(cfg.data.max_length_left,
                                     cfg.data.max_titles)

    train_order = np.random.permutation(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=32,
        prefetch_factor=8,
        collate_fn=collate_fn,
        batch_sampler=torch.utils.data.sampler.BatchSampler(
            MySampler(train_order),
            64 if cfg.test_mode else cfg.data.batch_size,
            False),
    )

    # breakpoint()
    return train_loader