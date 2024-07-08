import torch
import numpy as np


def self_attention(max_len, seq_len, dtype):
    mask = np.ones((12, max_len, max_len), dtype=dtype) * float("-inf")
    mask[:, :seq_len, :seq_len] = 0
    mask[:, seq_len:max_len, 0] = 0
    return mask


def cross_attention(max_len, seq_len, dtype):
    eff_len = min(max_len, seq_len)
    mask = np.ones((12, max_len, max_len), dtype=dtype) * float("-inf")
    mask[:, np.arange(eff_len), np.arange(eff_len)] = 0
    mask[:, :, 0] = 0
    return mask


class NGAMEPublicDataset(torch.utils.data.Dataset):
    def __init__(self, trn_X_Y, prefix, tokenization_folder, max_length_left, max_length_right, num_samples_test=-1, right_prefix="label", dtype="int64"):
        self.max_length_left = max_length_left
        self.max_length_right = max_length_right
        self.num_samples_test = num_samples_test

        self.labels = trn_X_Y
        self.X_am = np.memmap(f"{tokenization_folder}/{prefix}_attention_mask.dat", dtype=dtype, mode="r").reshape(-1, max_length_left)
        self.X_iid = np.memmap(f"{tokenization_folder}/{prefix}_input_ids.dat", dtype=dtype, mode="r").reshape(-1, max_length_left)

        self.Y_am = np.memmap(f"{tokenization_folder}/{right_prefix}_attention_mask.dat", dtype=dtype, mode="r").reshape(-1, max_length_right)
        self.Y_iid = np.memmap(f"{tokenization_folder}/{right_prefix}_input_ids.dat", dtype=dtype, mode="r").reshape(-1, max_length_right)

        if right_prefix != "label":
            self.Y_iid = np.vstack([self.X_iid, self.Y_iid])
            self.Y_am = np.vstack([self.X_am, self.Y_am])

    def __getitem__(self, index):
        pos_indices = self.labels[index].indices
        pos_ind = np.random.choice(pos_indices)

        return (self.X_iid[index], self.X_am[index], pos_indices, pos_ind, self.Y_iid[pos_ind], self.Y_am[pos_ind], index)

    def __len__(self):
        if self.num_samples_test == -1:
            return self.X_am.shape[0]
        else:
            return self.num_samples_test


class OAKPublicDataset(NGAMEPublicDataset):
    def __init__(
        self,
        trn_X_Y,
        prefix,
        tokenization_folder,
        max_length_left,
        max_length_right,
        max_titles,
        doc_meta_sp,
        dtype="int64",
        num_samples_test=-1,
        right_prefix="label",
        meta_prefix="graph",
    ):
        super(OAKPublicDataset, self).__init__(
            trn_X_Y, prefix, tokenization_folder, max_length_left, max_length_right, dtype=dtype, num_samples_test=num_samples_test, right_prefix=right_prefix
        )

        self.max_titles = max_titles
        self.doc_meta = doc_meta_sp
        self.meta_degree = self.doc_meta.getnnz(axis=1)
        self.dtype = dtype
        # self.attention_builder = self_attention
        self.attention_builder = cross_attention

    def __getitem__(self, index):
        item = super(OAKPublicDataset, self).__getitem__(index)

        titles = np.zeros(self.max_titles, dtype=self.dtype)
        titles_am = self.attention_builder(self.max_titles + 1, self.meta_degree[index] + 1, np.float32)
        titles[: min(self.meta_degree[index], self.max_titles)] = np.random.choice(
            self.doc_meta[index].indices, size=min(self.max_titles, self.meta_degree[index]), replace=False
        )
        meanpool_mask = np.zeros((1, self.max_titles + 1), dtype=self.dtype)
        meanpool_mask[0, : min(self.meta_degree[index], self.max_titles) + 1] = 1

        return (*item, titles, titles_am, meanpool_mask)


class OAKEncMetaPublicDataset(OAKPublicDataset):
    def __init__(
        self,
        trn_X_Y,
        prefix,
        tokenization_folder,
        max_length_left,
        max_length_right,
        max_titles,
        doc_meta_sp,
        dtype="int64",
        num_samples_test=-1,
        right_prefix="label",
        meta_prefix="graph",
    ):
        super(OAKEncMetaPublicDataset, self).__init__(
            trn_X_Y,
            prefix,
            tokenization_folder,
            max_length_left,
            max_length_right,
            max_titles,
            doc_meta_sp,
            dtype=dtype,
            num_samples_test=num_samples_test,
            right_prefix=right_prefix,
            meta_prefix=meta_prefix,
        )

        self.M_am = np.memmap(f"{tokenization_folder}/{meta_prefix}_attention_mask.dat", dtype=dtype, mode="r").reshape(-1, max_length_right)
        self.M_iid = np.memmap(f"{tokenization_folder}/{meta_prefix}_input_ids.dat", dtype=dtype, mode="r").reshape(-1, max_length_right)

    def __getitem__(self, index):
        item = super(OAKEncMetaPublicDataset, self).__getitem__(index)
        ttl_idx = item[-3]
        ttl_iid = self.M_iid[ttl_idx.reshape(1, -1)]
        ttl_am = self.M_am[ttl_idx.reshape(1, -1)]

        return (*item, ttl_iid, ttl_am)


class PRFPublicDataset(NGAMEPublicDataset):
    def __init__(self, trn_X_Y, prefix, tokenization_folder, max_length_prf, max_length_right, num_samples_test=-1, dtype="int64"):
        self.max_length_prf = max_length_prf
        self.max_length_right = max_length_right
        self.num_samples_test = num_samples_test

        self.labels = trn_X_Y
        self.X_am = np.memmap(f"{tokenization_folder}/{prefix}_prf_attention_mask.dat", dtype=dtype, mode="r").reshape(-1, max_length_prf)
        self.X_iid = np.memmap(f"{tokenization_folder}/{prefix}_prf_input_ids.dat", dtype=dtype, mode="r").reshape(-1, max_length_prf)

        self.Y_am = np.memmap(f"{tokenization_folder}/label_attention_mask.dat", dtype=dtype, mode="r").reshape(-1, max_length_right)
        self.Y_iid = np.memmap(f"{tokenization_folder}/label_input_ids.dat", dtype=dtype, mode="r").reshape(-1, max_length_right)
