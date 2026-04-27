import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import faiss

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
        features = self.features

        # # Compute pairwise distances
        # distances = torch.cdist(features, features)
        #
        # # Find indices of k nearest neighbors for each feature
        # _, nearest_indices = distances.topk(topk + 1, largest=False, dim=1)
        # k_nearest_neighbours = nearest_indices[:, 1:]  # exclude self as nearest neighbor
        #
        # # Find indices of k furthest neighbors for each feature
        # _, furthest_indices = distances.topk(topk, largest=True, dim=1)
        # k_furthest_neighbours = furthest_indices[:, :]

        # index = nmslib.init(method='hnsw', space='12')
        # index.addDataPointBatch(features)
        # index.createIndex({'post':2}, prin_progress = True)
        # ids , distances = index.knnQueryBatch(features, k=len(features), num_threads=4)
        #
        # k_furthest_neighbours = ids[:, -1:]
        # k_nearest_neighbours = ids[:, 1:]


        d = features.shape[1]
        index = faiss.IndexFlatL2(d)
        # index.add(features)
        index.add(features.cpu().numpy())  # CUDA

        xq = np.random.random(d)
        _, ids = index.search(xq.reshape(1, -1).astype(np.float32), len(features))
        sz = ids.shape[1]
        k_furthest_neighbours = ids.reshape(sz, 1)[::-1]
        k_nearest_neighbours = ids[:, :].reshape(sz, 1)

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
