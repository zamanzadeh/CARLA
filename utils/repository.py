import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

class TSRepository(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        features = self.features.cpu().numpy()
        knn_model = NearestNeighbors(n_neighbors=features.shape[0],
                                 algorithm='brute',
                                 n_jobs=-1)
        knn_model.fit(features)

        distances, indices = knn_model.kneighbors(features, return_distance=True)
        k_furthest_neighbours = []
        k_nearest_neighbours = []
        for i in range(features.shape[0]):
            # sort the neighbours based on their distance to the point
            sorted_indices = np.argsort(distances[i])
            # get the k furthest neighbours for each point
            k_furthest_neighbours.append(indices[i][sorted_indices[-topk:]])
            k_nearest_neighbours.append(indices[i][sorted_indices[1:topk+1]])

        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)

            return k_furthest_neighbours, k_nearest_neighbours, accuracy
        
        else:
            return k_furthest_neighbours, k_nearest_neighbours

    def furthest_nearest_neighbors(self, topk):
        """
        Compute per-sample k-nearest and k-furthest neighbors.

        Uses torch.cdist for pairwise distances (batched to avoid OOM on large datasets).
        Returns:
            k_furthest_neighbours: np.ndarray of shape (n, topk) — indices of furthest neighbors
            k_nearest_neighbours:  np.ndarray of shape (n, topk) — indices of nearest neighbors
        """
        features = self.features.cpu()  # (n, d)
        n = features.shape[0]
        BATCH = 2048  # process rows in chunks to keep memory bounded

        all_near = []
        all_far  = []

        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            batch = features[start:end]                        # (b, d)

            # Pairwise L2 distances between this batch and all samples
            with torch.no_grad():
                dists = torch.cdist(batch, features, p=2)     # (b, n)

            # --- k-nearest: set self-distance to inf so it is never picked ---
            for i in range(end - start):
                dists[i, start + i] = float('inf')
            _, near_ids = torch.topk(dists, topk, dim=1, largest=False)  # (b, topk)

            # --- k-furthest: restore self-distance to -inf so it is never picked ---
            for i in range(end - start):
                dists[i, start + i] = float('-inf')
            _, far_ids = torch.topk(dists, topk, dim=1, largest=True)    # (b, topk)

            all_near.append(near_ids.numpy())
            all_far.append(far_ids.numpy())

        k_nearest_neighbours  = np.vstack(all_near)   # (n, topk)
        k_furthest_neighbours = np.vstack(all_far)    # (n, topk)

        return k_furthest_neighbours, k_nearest_neighbours


    def reset(self):
        self.ptr = 0

    def resize(self, sz):
        self.n = sz * self.n
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        if not torch.is_tensor(targets): targets = torch.from_numpy(targets)
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')
