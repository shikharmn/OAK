import torch
from typing import Optional


class _Loss(torch.nn.Module):
    def __init__(self, reduction='mean', pad_ind=None):
        super(_Loss, self).__init__()
        self.reduction = reduction
        self.pad_ind = pad_ind

    def _reduce(self, loss):
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'custom':
            return loss.sum(dim=1).mean()
        else:
            return loss.sum()

    def _mask_at_pad(self, loss):
        """
        Mask the loss at padding index, i.e., make it zero
        """
        if self.pad_ind is not None:
            loss[:, self.pad_ind] = 0.0
        return loss

    def _mask(self, loss, mask=None):
        """
        Mask the loss at padding index, i.e., make it zero
        * Mask should be a boolean array with 1 where loss needs
        to be considered.
        * it'll make it zero where value is 0
        """
        if mask is not None:
            loss = loss.masked_fill(~mask, 0.0)
        return loss


class TripletMarginLossOHNM(_Loss):
    r""" Triplet Margin Loss with Online Hard Negative Mining

    * Applies loss using the hardest negative in the mini-batch
    * Assumes diagonal entries are ground truth (for multi-class as of now)

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    margin: float, optional (default=0.8)
        margin in triplet margin loss
    k: int, optional (default=2)
        compute loss only for top-k negatives in each row 
    apply_softmax: boolean, optional (default=2)
        promotes hard negatives using softmax
    """

    def __init__(self, reduction='mean', margin=0.8, k=3, apply_softmax=False, tau=0.1, num_violators=False):
        super(TripletMarginLossOHNM, self).__init__(reduction=reduction)
        self.margin = margin
        self.k = k
        self.tau = tau
        self.num_violators = num_violators
        self.apply_softmax = apply_softmax

    def forward(self,
                input: torch.FloatTensor,
                target: torch.FloatTensor,
                mask: Optional[torch.BoolTensor] = None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is False

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        sim_p = torch.diagonal(input).view(-1, 1)
        similarities = torch.where(
            target == 0, input, torch.full_like(input, -50))
        _, indices = torch.topk(similarities, largest=True, dim=1, k=self.k)
        sim_n = input.gather(1, indices)
        loss = torch.max(torch.zeros_like(sim_p), sim_n - sim_p + self.margin)
        if self.apply_softmax:
            sim_n[loss == 0] = -50
            prob = torch.softmax(sim_n / self.tau, dim=1)
            loss = loss * prob
        reduced_loss = self._reduce(loss)
        if self.num_violators:
            # nnz = torch.sum((loss > 0), axis=1).float().mean()
            return reduced_loss, torch.sum((loss > 0), axis=1)
        else:
            return reduced_loss


class MultimodalConfidenceLoss(_Loss):
    def __init__(self, reduction='mean', margin=0.2, k=2):
        super(MultimodalConfidenceLoss, self).__init__(reduction=reduction)
        self.margin = margin
        self.num_negatives = k
        
    def forward(self, enc_scores, comb_scores, gt):
        viol_conf = (2 * gt - 1) * (enc_scores - comb_scores)
        _, _ind = torch.topk(viol_conf, dim=1, k=self.num_negatives)
        top_viol_conf = viol_conf.gather(1, _ind)
        loss = torch.max(torch.zeros_like(top_viol_conf), top_viol_conf + self.margin)
        return self._reduce(loss), torch.sum((loss > 0), axis=1)
    