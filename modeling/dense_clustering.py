import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import time
import numpy as np
import torch.nn.functional as F
from multiprocessing import Pool
import torch.multiprocessing as mp
import pickle
import colorama

colorama.init()
COLORS = {0: colorama.Fore.BLUE,
          1: colorama.Fore.CYAN,
          2: colorama.Fore.GREEN,
          3:colorama.Fore.MAGENTA,
          4: colorama.Fore.RED,
          5: colorama.Fore.WHITE,
          6: colorama.Fore.YELLOW,
          7: colorama.Fore.LIGHTRED_EX,
          8: colorama.Fore.LIGHTMAGENTA_EX,
          9: colorama.Fore.LIGHTBLUE_EX,
          10: colorama.Fore.LIGHTCYAN_EX,
          11: colorama.Fore.LIGHTGREEN_EX,
          12: colorama.Fore.LIGHTYELLOW_EX,
          13: colorama.Fore.LIGHTWHITE_EX}


def _multi_gpu_mm(return_dict, rank, a, b):
    return_dict[rank] = torch.mm(a, b).cpu()

def multi_gpu_mm(a_chunks, bs):
    print("---inside multi_gpu_mm")
    processes = []
    manager = mp.Manager()
    return_dict = manager.dict()
    for c in range(len(a_chunks)):
        p = mp.Process(target=_multi_gpu_mm, args=(return_dict, c, a_chunks[c], bs[c]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    keys = sorted(list(return_dict.keys()))
    ret = torch.vstack([return_dict[k] for k in keys])

    return ret

def split_cluster(labels_features, indices, devices=None, metric='cosine', tol=1e-4, leakage=None):
    """
    If devices is not None, split label_features on devices to calculate similarity
    """
    m = labels_features.shape[0]
    if(devices is not None):
        chunk_size = int(np.ceil(m / len(devices)))
        a_chunks = [labels_features[i * chunk_size: (i + 1) * chunk_size].to(devices[i]) for i in range(len(devices))]
    
    with torch.no_grad():
        n = labels_features.shape[0]
        if labels_features.shape[0] == 1:
            return [indices]
        cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))

        while cluster[0] == cluster[1]:
            cluster = np.random.randint(
                low=0, high=labels_features.shape[0], size=(2))
        _centeroids = labels_features[cluster]

        _similarity = torch.mm(labels_features, _centeroids.T)
        old_sim, new_sim = -1000000, -2

        while new_sim - old_sim >= tol:
            clustered_lbs = torch.split(torch.argsort(_similarity[:, 1]-_similarity[:, 0]), (_similarity.shape[0]+1) // 2)
            _centeroids = F.normalize(torch.vstack([torch.mean(labels_features[x, :], axis=0) for x in clustered_lbs]))
            if(devices is None):  # assume labels_features already on correct device
                _similarity = torch.mm(labels_features, _centeroids.T)
            else:  # do parallelized mm 
                b = _centeroids.T
                bs = [b.to(devices[i]) for i in range(len(devices))]
                _similarity = multi_gpu_mm(a_chunks, bs)
            
            old_sim, new_sim = new_sim, sum([torch.sum(_similarity[indx, i]) for i, indx in enumerate(clustered_lbs)]).item() / n
            #print(new_sim)
        del _similarity

        return list(map(lambda x: indices[x], clustered_lbs))

def balanced_cluster_gpu(return_dict, rank, embs, num_levels, device, verbose):
    embs = embs.to(device)
    m = embs.shape[0]
    clusters = [torch.arange(m).to(device)]
    for t in range(num_levels):
        start = time.time()
        new_clusters = []
        for cluster in clusters:
            new_clusters += split_cluster(embs[cluster], cluster)
        clusters = new_clusters
        end = time.time()
        if(verbose):
            print(COLORS[rank % 8], f"rank={rank} => Total clusters {len(clusters)}\tAvg. Cluster size {'%.2f'%(np.mean([len(x) for x in clusters]))}\tTime to split nodes on this level {'%.2f'%(end-start)} sec")
    
    del embs
    torch.cuda.empty_cache()

    return_dict[rank] = [cluster.cpu().numpy() for cluster in clusters]

def balanced_cluster_gpu_dist(return_dict, rank, embs, num_levels, device, verbose):
    embs = embs.to(device)
    m = embs.shape[0]
    clusters = [torch.arange(m)]
    for t in range(num_levels):
        start = time.time()
        multi_pool = Pool(processes=5)
        ret = multi_pool.starmap(split_cluster, [(embs[cluster], cluster) for cluster in clusters])
        multi_pool.close() 
        multi_pool.join()
        end = time.time()
        clusters = [x for subl in ret for x in subl]
        if(verbose):
            print(f"rank={rank} => Total clusters {len(clusters)}\tAvg. Cluster size {'%.2f'%(np.mean([len(x) for x in clusters]))}\tTime to split nodes on this level {'%.2f'%(end-start)} sec")

    del embs
    torch.cuda.empty_cache()

    return_dict[rank] = clusters

def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)

def next_power_of_two(target):
    if target > 1:
        for i in range(1, int(target)):
            if (2 ** i >= target):
                return 2 ** i
    else:
        return 1

def balanced_cluster(embs, num_levels, devices, num_random_clusters=-1, verbose=True):
    num_devices = len(devices)
    m = embs.shape[0]

    if(num_devices == 1):  # if only 1 device, assume public dataset
        num_random_clusters = -1

    if(num_random_clusters > 0):
        assert is_power_of_two(num_random_clusters), "Clusters to create randomly in balanced_cluster should be power of 2"
        print("doing random split")
        random_indices = np.random.permutation(m)
        lengths = [m]
        while(len(lengths) != num_random_clusters):
            new_lengths = []
            for length in lengths:
                new_lengths.append(int(np.floor(length / 2)))
                new_lengths.append(int(np.ceil(length / 2)))
            lengths = new_lengths
        print("lengths:", lengths)
        assert sum(lengths) == m
        clusters = np.split(random_indices, np.cumsum(lengths)[:-1])
        clusters = [torch.LongTensor(cluster) for cluster in clusters]
    else:
        num_cpu_clusters = next_power_of_two(num_devices)
        print("doing cpu split")
        clusters = [torch.arange(m)]
        while(len(clusters) < num_cpu_clusters):
            start = time.time()
            if(len(clusters) == 1):
                new_clusters = []
                for cluster in clusters:
                    new_clusters += split_cluster(embs[cluster], cluster)
                    # new_clusters += split_cluster(embs[cluster], cluster, devices)  # <== slower

                clusters = new_clusters
            else:
                multi_pool = Pool(processes=10)
                ret = multi_pool.starmap(split_cluster, [(embs[cluster], cluster) for cluster in clusters])
                multi_pool.close() 
                multi_pool.join()
                end = time.time()
                clusters = [x for subl in ret for x in subl]
            end = time.time()
            if(verbose):
                print(f"Total clusters {len(clusters)}\tAvg. Cluster size {'%.2f'%(np.mean([len(x) for x in clusters]))}\tTime to split nodes on this level {'%.2f'%(end-start)} sec")
    
    remaining_levels = int(num_levels - np.log2(len(clusters)))
    if(verbose):
        print(f"remaining levels for GPU split={remaining_levels}")
    
    processes = []
    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    return_dict = manager.dict()
    for i in range(0, len(clusters), num_devices):
        print(f"==> gpu splitting random clusters {i} to {min(i + num_devices, len(clusters))}")
        for j in range(num_devices):
            if(i + j >= len(clusters)):
                break
            p = ctx.Process(target=balanced_cluster_gpu, args=(return_dict, i + j, embs[clusters[i + j]], remaining_levels, devices[j], verbose))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    
    clusters = [cluster.cpu().numpy() for cluster in clusters]
    final_clusters = []
    gpu_cluster_ids = sorted(return_dict.keys())
    for j in gpu_cluster_ids:
        gpu_clusters = return_dict[j]
        for gpu_cluster in gpu_clusters:
            final_clusters.append(clusters[j][gpu_cluster])

    assert len(set([x for clus in final_clusters for x in clus])) == embs.shape[0], "issue in clustering, clusters not mutually exclusive or exhaustive"
    
    print(colorama.Style.RESET_ALL)
    return final_clusters

def update_clustering(cfg, emb_bank, train_loader, clusters=None):
    if cfg.epoch in cfg.data.curr_steps:
        cfg.data.cl_size *= 2
        print(f"Changing cluster size to {cfg.data.cl_size}")

    if cfg.epoch >= cfg.data.cl_start:
        if (cfg.epoch - cfg.data.cl_start) % cfg.data.cl_update == 0:
            print(
                f"Updating clusters with cluster size {cfg.data.cl_size} (using stale embeddings)")
            embs = emb_bank.copy()
            tree_depth = int(
                np.ceil(np.log(embs.shape[0] / cfg.data.cl_size) / np.log(2)))
            print(f"tree depth = {tree_depth}")
            if len(cfg.clustering_devices) > 1:
                num_random_clusters = cfg.data.num_random_clusters if cfg.data.num_random_clusters != - \
                    1 else next_power_of_two(len(cfg.clustering_devices))
                if num_random_clusters < len(cfg.clustering_devices):
                    clusters = balanced_cluster(torch.HalfTensor(embs),
                                                tree_depth,
                                                cfg.clustering_devices,
                                                num_random_clusters,
                                                True)
            else:
                clusters = balanced_cluster(torch.HalfTensor(embs),
                                            tree_depth,
                                            cfg.clustering_devices,
                                            -1,
                                            True)
            del embs
            gc.collect()

        random_cluster_order = np.random.permutation(len(clusters))
        random_point_order = [clusters[x] for x in random_cluster_order]
        random_point_order = [x for subl in random_point_order for x in subl]
        train_loader.batch_sampler.sampler.update_order(random_point_order)
    else:
        train_loader.batch_sampler.sampler.update_order(
            np.random.permutation(len(train_loader.dataset)))
        
    return clusters
