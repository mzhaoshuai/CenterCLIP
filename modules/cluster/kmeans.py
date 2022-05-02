#coding=utf-8
"""
An implementation of K-medoids in PyTorch (loop implementation)

Reference
[1] https://github.com/subhadarship/kmeans_pytorch
"""
import torch
from .cluster_utils import pairwise_distance, KKZ_init


@torch.no_grad()
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def batch_kmedoids(X, K, distance='euclidean', threshold=1e-5, iter_limit=60,
					id_sort=True, batch_distance=False, norm_p=2.0):
	"""
	perform batch kmeans
	Args:
		X: (torch.tensor) matrix, dtype should be torch.float
		K: (int) number of clusters
		distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
		threshold: (float) threshold [default: 0.0001]
		iter_limit: hard limit for max number of iterations
		id_sort: whether sort id of cluster centers in ascending order 

	Return:
		(cluster_assginment, mediods)
	"""
	assert distance in ['euclidean', 'cosine'] and X.ndim == 3

	B, N, L = X.shape[0], X.shape[1], X.shape[2]
	if batch_distance:
		# [B, N, N]
		distance_matrix = pairwise_distance(X, X, metric=distance, self_nearest=True, p=norm_p)					
	else:
		distance_matrix = None
	# import pdb; pdb.set_trace()
	medoids_list, ids_list = [], []
	for b in range(B):
		cluster_assginment, mediods = kmedoids(X[b, ...], K,
												distance_matrix[b, ...] if batch_distance else None,
												metric=distance,
												threshold=threshold, iter_limit=iter_limit,
												id_sort=id_sort)
		medoids_list.append(mediods)
		ids_list.append(cluster_assginment)

	return torch.stack(ids_list, dim=0), torch.stack(medoids_list, dim=0)


@torch.no_grad()
def kmedoids(X, K, distance_matrix=None, threshold=1e-4, iter_limit=50,
				metric='euclidean', id_sort=True):
	"""
	perform k mediods
	Args:
		X: (torch.tensor) matrixm, dtype should be torch.float
		K: (int) number of clusters
		distance_matrix: torch.Tensor, pairwise distance matrix of input
		threshold: (float) threshold [default: 0.0001]
		iter_limit: hard limit for max number of iterations
		id_sort: whether sort id of cluster centers in ascending order 

	Return:
		(cluster_assginment, mediods)
	"""
	assert X.ndim == 2
	if distance_matrix is None:
		distance_matrix = pairwise_distance(X, X, metric=metric, self_nearest=True)
	
	# step 1: initialize medoids
	# mediods = kmeans_plusplus_init(distance_matrix, K)						# [K]
	mediods = KKZ_init(X, distance_matrix, K, batch=False)
	# print('mediods:\t', mediods)

	for step in range(iter_limit):
		# step 2: assign points to medoids
		pre_mediods = mediods
		sub_dis_matrix = distance_matrix[:, mediods]							# [N, K]
		min_dis, cluster_assginment = torch.min(sub_dis_matrix, dim=-1)			# [N]
		
		# step 3: update medoids
		for i in range(K):
			mask = (cluster_assginment == i)
			mask_indices = mask.nonzero()
			# shape [K_i, K_i]
			ki_dis_matrix = distance_matrix[mask, :][:, mask]
			ki_row_sum = torch.sum(ki_dis_matrix, dim=-1)
			if ki_row_sum.numel() < 1:
				print('pre_mediods:\t', pre_mediods)
				print('step:\t', step)
				print('cluster:\t', i)
				print('cluster_assginment:\t', cluster_assginment)
				print('ki_dis_matrix:\t', ki_dis_matrix)
				print('ki_row_sum:\t', ki_row_sum)
				raise ValueError("the distance matrix (ki_row_sum) cannot be empty", ki_row_sum)
			ki_ind = torch.argmin(ki_row_sum)
			mediods[i] = mask_indices[ki_ind]

		center_shift = torch.sum(
							torch.sqrt(
								torch.sum((X[mediods, :] - X[pre_mediods, :]) ** 2, dim=-1)
							)
						)
		if center_shift < threshold:
			break

	if id_sort:
		mediods, _ = torch.sort(mediods)
		# step 2: assign points to medoids
		sub_dis_matrix = distance_matrix[:, mediods]						# [N, K]
		min_dis, cluster_assginment = torch.min(sub_dis_matrix, dim=-1)		# [N]

	return cluster_assginment, mediods


if __name__ == "__main__":
	a1, a2, a3 = torch.ones(2, 2), torch.ones(4, 2) * 5, torch.ones(6, 2) * 9
	a4 = torch.ones(3, 2) * 12
	X = torch.cat([a1, a2, a3, a4], dim=0)
	idx = torch.randperm(X.shape[0])
	X = X[idx, :]
	print(X)
	distance_matrix = pairwise_distance(X, X, metric='euclidean')
	assign, ids = kmedoids(X, 3, distance_matrix, threshold=1e-4, iter_limit=30)
	print(assign, ids)
	X_id = torch.cat([X, assign.unsqueeze(1)], dim=1)
	print(X_id)
	print(X[ids])

	# x = torch.rand(2, 7, 3)
	# assign, ids = batch_kmedoids(x, K=2, id_sort=False)
	# print(x)
	# print('assign:\n', assign)
	# print('ids:\n', ids)

	# assign, ids = batch_kmedoids(x, K=3, id_sort=True)
	# print('\n')
	# print('(sorted) assign:\n', assign)
	# print('(sorted) ids:\n', ids)
	# import pdb; pdb.set_trace()
	# cluster = x[torch.arange(x.shape[0]).unsqueeze(-1), ids]
	# print(cluster, cluster.shape)
	# test kmeans++ initilization
	# X = torch.rand(5, 4)
	# distance_matrix = pairwise_distance(X, X)
	# mediods = kmeans_plusplus_init(distance_matrix, K=3)
