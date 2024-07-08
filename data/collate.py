import torch
import numpy as np

from collections import OrderedDict

def clip_batch_lengths(ind, mask, max_len):
    _max = min(np.max(np.sum(mask, axis=1)), max_len)
    return ind[:, :_max], mask[:, :_max]


class BaseCollateClass:
    """
    Basic collate function for providing the correctly sampled documents, labels and the batch selection matrix.
    """
    
    def __call__(self, batch):
        batch_labels = []
        random_pos_indices = []
        for item in batch:
            batch_labels.append(item[2])
            random_pos_indices.append(item[3])

        batch_size = len(batch_labels)
        
        ip_ind = np.vstack([x[0] for x in batch])
        ip_mask = np.vstack([x[1] for x in batch])
        op_ind = np.vstack([x[4] for x in batch])
        op_mask = np.vstack([x[5] for x in batch])
        
        batch_selection = np.zeros((batch_size, batch_size), dtype=np.float32)

        random_pos_indices_set = set(random_pos_indices)
        random_pos_indices = np.array(random_pos_indices, dtype=np.int32)

        for (i, item) in enumerate(batch_labels):
            intersection = set(item).intersection(random_pos_indices_set)
            result = np.zeros(batch_size, dtype=np.float32)
            for idx in intersection:
                result += (idx == random_pos_indices)
            batch_selection[i] = result
        
        return ip_ind, ip_mask, op_ind, op_mask, batch_selection

class NGAMECollateClass(BaseCollateClass):
    """
    Collate class for NGAME that inherits from BaseCollateClass on handling documents and labels.
    """
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, batch):
        (ip_ind, ip_mask,
        op_ind, op_mask,
        batch_selection) = super(NGAMECollateClass, self).__call__(batch)
        
        lbl_ind = np.vstack([x[6] for x in batch])    
        batch_data = OrderedDict()
        
        batch_data['indices'] = torch.LongTensor([item[6] for item in batch])
        batch_data['ip_ind'] = torch.from_numpy(ip_ind)
        batch_data['ip_mask'] = torch.from_numpy(ip_mask)
        batch_data['op_ind'] = torch.from_numpy(op_ind)
        batch_data['op_mask'] = torch.from_numpy(op_mask)
        batch_data['Y'] = torch.from_numpy(batch_selection)
        batch_data['lbl_ind'] = torch.LongTensor(lbl_ind.ravel())
        
        return batch_data


class OAKCollateClass(NGAMECollateClass):
    """
    Collate class for OAK that inherits from NGAMECollateClass on how to handle everything except the metadata
    indices.
    """
    def __init__(self, max_length, max_titles):
        super(OAKCollateClass, self).__init__(max_length)

    def __call__(self, batch):
        batch_data = super(OAKCollateClass, self).__call__(batch)
        ttl_ind = np.vstack([x[7] for x in batch])
        ttl_mask = np.vstack([x[8] for x in batch])
        mean_mask = np.vstack([x[9] for x in batch])
        
        batch_data['ttl_ind'] = torch.from_numpy(ttl_ind)
        batch_data['ttl_mask'] = torch.from_numpy(ttl_mask)
        batch_data['mean_mask'] = torch.from_numpy(mean_mask)
    
        return batch_data
