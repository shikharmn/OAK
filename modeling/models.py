import torch
import numpy as np
import torch.nn as nn
import sentence_transformers
import torch.nn.functional as F


class CustomSTEmbedding(nn.Module):
    """
    Wrapper to obtain bert embeddings from ST models.
    """

    def __init__(self, bert):
        super(CustomSTEmbedding, self).__init__()
        if isinstance(bert, str):
            self.bert = sentence_transformers.SentenceTransformer(bert)
        else:
            self.bert = bert

    def forward(self, input_ids, attention_mask):
        return self.bert({"input_ids": input_ids, "attention_mask": attention_mask})["sentence_embedding"]


class BERTEmbedding(nn.Module):
    def __init__(self, encoder_name, transform_dim, encoder_type="st", vocab_size=30522):
        super(BERTEmbedding, self).__init__()

        if encoder_type == "st":
            print(f"Using a st encoder {encoder_name}")
            self.encoder = CustomSTEmbedding(encoder_name)
        else:
            raise NotImplementedError("Invalid encoder type")

        self.transform_dim = transform_dim

        if self.transform_dim > 0:
            self.transform = nn.Linear(768, self.transform_dim)

    def forward(self, input_ids, attention_mask):
        if self.transform_dim > 0:
            return self.transform(self.encoder(input_ids, attention_mask))
        else:
            return self.encoder(input_ids, attention_mask)

    @property
    def repr_dims(self):
        return 768 if self.transform_dim < 0 else self.transform_dim


def Projection(num_embeddings, embedding_dim):
    linear = nn.Linear(num_embeddings, embedding_dim)
    if num_embeddings == embedding_dim:
        torch.nn.init.eye_(linear.weight)
    else:
        torch.nn.init.kaiming_uniform_(linear.weight)
    linear.bias.data.fill_(0)
    linear = nn.utils.spectral_norm(linear)
    return linear


class CrossGatingCombiner(nn.Module):
    def __init__(self, n_dim) -> None:
        super(CrossGatingCombiner, self).__init__()
        self.n_dim = n_dim
        self.drop = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(n_dim)
        self.activation = nn.Softmax(dim=1)
        self.doc_proj = Projection(n_dim, n_dim)
        self.meta_proj = Projection(n_dim, n_dim)
        self.out_proj = Projection(n_dim, n_dim)

    def forward(self, u, v, mean_mask):
        # Ready the projected representations
        attention_mask = mean_mask[:, 1:].float()
        meta_absent = torch.where(attention_mask.sum(dim=-1) == 0)[0]
        attention_mask[meta_absent, 0] = 1.0
        bsz, m_sq, n_dim = v.size()
        doc = self.doc_proj(u) / np.sqrt(self.n_dim)
        meta = self.meta_proj(v.view(bsz * m_sq, n_dim)).view(bsz, m_sq, n_dim)

        # Calculate gating scores
        scores = (doc.view(bsz, 1, n_dim) * meta).sum(dim=-1)  # (bsz, m_sq)
        scores.masked_fill_(attention_mask == 0, float("-inf"))  # Attention masking
        scores = self.activation(scores)
        scores = self.drop(scores)

        # Obtain the fused representations
        attended_meta = scores.view(bsz, 1, m_sq) @ v  # (bsz, 1, n_dim) Combining attended metadata vectors
        projattn_meta = self.out_proj(attended_meta.view(bsz, n_dim))
        fused_doc = self.norm(u + projattn_meta)

        # Replacing where no metadata with encoder representation FIX HACK
        mean_mask = attention_mask
        mean_mask[meta_absent, 0] = 1.0
        fused_doc[meta_absent] = u[meta_absent]
        return F.normalize(fused_doc)
    

class NGAMENetwork(nn.Module):
    def __init__(self, encoder_name, encoder_type, transform_dim, device="cpu", vocab_size=30522):
        super(NGAMENetwork, self).__init__()
        self.encoder = BERTEmbedding(encoder_name, transform_dim, encoder_type=encoder_type, vocab_size=vocab_size)
        self.device = device

    def encode(self, doc_input_ids, doc_attention_mask):
        return F.normalize(self.encoder(doc_input_ids, doc_attention_mask))

    def encode_document(self, doc_input_ids, doc_attention_mask, *args):
        return F.normalize(self.encoder(doc_input_ids, doc_attention_mask))

    def encode_label(self, lbl_input_ids, lbl_attention_mask):
        u = self.encoder(lbl_input_ids, lbl_attention_mask)
        return F.normalize(u)

    def forward(self, doc_input_ids, doc_attention_mask, lbl_input_ids, lbl_attention_mask):
        if doc_input_ids is None:
            return self.encode_label(lbl_input_ids, lbl_attention_mask)
        elif lbl_input_ids is None:
            return self.encode_document(doc_input_ids, doc_attention_mask)
        else:
            doc_embeddings = self.encode_document(doc_input_ids, doc_attention_mask)
            label_embeddings = self.encode_label(lbl_input_ids, lbl_attention_mask)
            return doc_embeddings, label_embeddings

    @property
    def repr_dims(self):
        return self.encoder.repr_dims


class OAKNetworkReg(nn.Module):
    def __init__(self, encoder_name, encoder_type, transform_dim, n_titles, device, vocab_size=30522, sparse=True):
        super(OAKNetworkReg, self).__init__()
        self.encoder = BERTEmbedding(encoder_name, transform_dim, encoder_type=encoder_type, vocab_size=vocab_size)
        self.embs = self._build_embedding_table(n_titles, sparse=sparse)
        self.device = device
        self.transform_dim = transform_dim
        self.combiner = CrossGatingCombiner(self.encoder.repr_dims)

    def encode_document(self, doc_input_ids, doc_attention_mask, meta_ids, mean_mask):
        """Obtain document side representations from encoder and combine them with AKP free parameters."""
        q_rep = F.normalize(self.encoder(doc_input_ids.to(self.device), doc_attention_mask.to(self.device)), dim=-1)
        t_reps = F.normalize(self.embs(meta_ids.to(self.device)).to(self.device), dim=-1)

        return self.combiner(q_rep, t_reps, mean_mask.bool().to(self.device)), q_rep

    def encode_label(self, lbl_input_ids, lbl_attention_mask):
        """Obtain label side representations from encoder."""
        return F.normalize(self.encoder(lbl_input_ids.to(self.device), lbl_attention_mask.to(self.device)))

    def forward(self, doc_input_ids, doc_attention_mask, lbl_input_ids, lbl_attention_mask, meta_ids=None, mean_mask=None):
        if doc_input_ids is None:
            return self.encode_label(lbl_input_ids, lbl_attention_mask)
        elif lbl_input_ids is None:
            return self.encode_document(doc_input_ids, doc_attention_mask, meta_ids, mean_mask)[0]
        else:
            doc_embeddings, doc_rep = self.encode_document(doc_input_ids, doc_attention_mask, meta_ids, mean_mask)
            label_embeddings = self.encode_label(lbl_input_ids, lbl_attention_mask)
            return doc_embeddings, label_embeddings, doc_rep
        # doc_embeddings = self.encode_document(doc_input_ids, doc_attention_mask, meta_ids, meta_mask)
        # return doc_embeddings

    def _build_embedding_table(self, n_titles, sparse=True):
        embs = nn.Embedding(n_titles, embedding_dim=self.encoder.repr_dims, sparse=sparse)
        nn.init.xavier_uniform_(embs.weight)
        return embs

    @property
    def repr_dims(self):
        return self.encoder.repr_dims