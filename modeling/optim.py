import torch
import transformers

class MultipleOptimizer(object):
    def __init__(self,
                 op,
                 gradient_scaling):
        self.optimizers = op
        self.grad_scaling = gradient_scaling

    def zero_grad(self,
                  *args, **kwargs):
        for op in self.optimizers:
            op.zero_grad(*args, **kwargs)

    def step(self,
             *args, **kwargs):
        for op in self.optimizers:
            op.step(*args, **kwargs)
    
    def scaled_step(self,
                    scaler,
                    *args, **kwargs):
        for op in self.optimizers:
            scaler.step(op, *args, **kwargs)

class MultipleScheduler(object):
    def __init__(self, sched):
        self.schedulers = sched
    
    def step(self, *args, **kwargs):
        for sched in self.schedulers:
            sched.step(*args, **kwargs)
            
def prepare_optimizer_and_scheduler(cfg, snet, t_total):
    """
    Set-up the optimizer and scheduler

    * t_total has to be pre-calculated (lr will be zero after these many steps)
    """
    no_decay = ['bias', 'LayerNorm.weight']
    t_total = t_total * cfg.data.epochs
    if cfg.ngame:
        gp = [
            {'params': [p for n, p in snet.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in snet.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        
        optimizer = MultipleOptimizer([torch.optim.AdamW(gp, **{'lr': cfg.data.lr, 'eps': 1e-06})], gradient_scaling=True)
        scheduler = MultipleScheduler([transformers.get_linear_schedule_with_warmup(optimizer.optimizers[0], num_warmup_steps=100, num_training_steps=t_total)])
        
        return optimizer, scheduler
    else:
        dense, sparse = [], []
        for k, p in snet.named_parameters():
            if p.requires_grad:
                if "embs" not in k:
                    dense.append((k,p))
                else:
                    sparse.append(p)
    
        gp = [
            {'params': [p for n, p in dense if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in dense if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]
    
    optimizer_list = [torch.optim.AdamW(gp, **{'lr': cfg.data.lr, 'eps': 1e-6}),
                      torch.optim.SparseAdam(sparse, **{'lr': cfg.data.lr * cfg.data.fp_lr_coeff, 'eps': 1e-6})]
    optimizer = MultipleOptimizer(optimizer_list, gradient_scaling=True)
    scheduler_list = [transformers.get_linear_schedule_with_warmup(optimizer.optimizers[0], num_warmup_steps=cfg.data.enc_lr_warmup, num_training_steps=t_total),
                        transformers.get_cosine_schedule_with_warmup(optimizer.optimizers[1], num_warmup_steps=cfg.data.fp_lr_warmup, num_training_steps=t_total)]
    
    scheduler = MultipleScheduler(scheduler_list)
    
    return optimizer, scheduler