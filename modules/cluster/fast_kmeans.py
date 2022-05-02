# coding=utf-8
"""
An implementation of fast k-means, acceleration comes from batch operations

Reference:
	[1] https://github.com/DeMoriarty/fast_pytorch_kmeans/blob/master/fast_pytorch_kmeans/kmeans.py
"""
import torch
from .cluster_utils import pairwise_distance, KKZ_init


@torch.no_grad()
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def batch_fast_kmedoids_with_split(X, K, distance='euclidean', threshold=1e-5, iter_limit=60,
									id_sort=True, norm_p=2.0, split_size=4, pre_norm=False):
	"""
	split a batch tensor into multiple chunks in order to avoid OOM of GPU mem
	Args:
		pre_norm: if true, do l2 normalization first before clustering
	"""
	if pre_norm:
		X = X / (X.norm(dim=-1, keepdim=True) + 1e-6)

	if X.shape[0] > split_size:
		all_t = torch.split(X, split_size, dim=0)
		assign_l, medoids_l = [], []
		for x_tmp in all_t:
			assign, medoids = batch_fast_kmedoids(x_tmp, K, distance=distance, threshold=threshold,
													iter_limit=iter_limit,
													id_sort=id_sort, norm_p=norm_p)
			assign_l.append(assign)
			medoids_l.append(medoids)

		return torch.cat(assign_l, dim=0), torch.cat(medoids_l, dim=0)

	else:
		assign, medoids = batch_fast_kmedoids(X, K, distance=distance, threshold=threshold,
												iter_limit=iter_limit,
												id_sort=id_sort, norm_p=norm_p)		
		return assign, medoids


@torch.no_grad()
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def batch_fast_kmedoids(X, K, distance='euclidean', threshold=1e-5, iter_limit=60,
						id_sort=True, norm_p=2.0):
	"""
	perform batch k mediods
	Args:
		X: (torch.tensor) matrixm, dtype should be torch.float
		K: (int) number of clusters
		distance_matrix: torch.Tensor, pairwise distance matrix of input
		threshold: (float) threshold [default: 0.0001]
		iter_limit: hard limit for max number of iterations
		id_sort: whether sort id of cluster centers in ascending order 
		norm_p: the norm of distance metric
	Return:
		(cluster_assginment, mediods)
	"""		
	assert distance in ['euclidean', 'cosine'] and X.ndim == 3

	B, N, L = X.shape[0], X.shape[1], X.shape[2]
	distance_matrix = pairwise_distance(X, X, metric=distance, all_negative=True,
											self_nearest=True, p=norm_p)					
	repeat_dis_m = distance_matrix.unsqueeze(1).repeat(1, K, 1, 1)							# [B, K, N, N]
	# step 1: initialize medoids (KKZ)						
	mediods = KKZ_init(X, distance_matrix, K, batch=True)   								# [B, K]
	batch_i = torch.arange(X.shape[0], dtype=torch.long, device=X.device).unsqueeze(1)		# [B, 1]
	# [B, K, 1]
	K_index = torch.arange(K, dtype=torch.long, device=X.device).reshape(1, K, 1).repeat(B, 1, 1)			

	for step in range(iter_limit):
		# step 2: assign points to medoids
		pre_mediods = mediods
		sub_dis_matrix = distance_matrix[batch_i, mediods, :]								# [B, K, N]
		min_dis, cluster_assginment = torch.min(sub_dis_matrix, dim=1)						# [B, N]

		# step 3: compute medoids
		cluster_assgin_r = cluster_assginment.unsqueeze(1).repeat(1, K, 1)					# [B, K, N]
		mask = (cluster_assgin_r == K_index)												# [B, K, N]
		sub_matrix = repeat_dis_m * mask.unsqueeze(-1) * mask.unsqueeze(-2)					# [B, K, N, N]
		mediods = torch.argmin(torch.sum(sub_matrix, dim=-1), dim=-1)						# [B, K]

		# the shift of mediods
		center_shift = torch.sum((X[batch_i, mediods, :] - X[batch_i, pre_mediods, :]) ** 2,
									dim=-1).sqrt().sum(dim=-1).mean()
		if center_shift < threshold:
			break

	if id_sort:
		mediods, _ = torch.sort(mediods, dim=1)
		# step 2: assign points to medoids
		sub_dis_matrix = distance_matrix[batch_i, mediods, :]								# [B, K, N]
		min_dis, cluster_assginment = torch.min(sub_dis_matrix, dim=1)						# [B, N]

	# print('The step is {}'.format(step))
	return cluster_assginment, mediods


@torch.no_grad()
def fast_kmedoids(X, K, distance_matrix=None, threshold=1e-5, iter_limit=50,
					metric='euclidean', id_sort=True, norm_p=2.0):
	"""
	perform k mediods
	Args:
		X: (torch.tensor) matrixm, dtype should be torch.float
		K: (int) number of clusters
		distance_matrix: torch.Tensor, pairwise distance matrix of input
		threshold: (float) threshold [default: 0.0001]
		iter_limit: hard limit for max number of iterations
	Return:
		(cluster_assginment, mediods)
	"""
	assert X.ndim == 2
	N, D = X.shape
	if distance_matrix is None:
		distance_matrix = pairwise_distance(X, X, metric=metric, all_negative=True,
												self_nearest=True, p=norm_p)
	repeat_dis_m = distance_matrix.unsqueeze(0).repeat(K, 1, 1)						# [K, N, N]
	# step 1: initialize medoids (kmeans++)						
	mediods = KKZ_init(X, distance_matrix, K)                               		# [K]
	K_index = torch.arange(K, dtype=torch.long, device=X.device).unsqueeze(-1)		# [K, 1]

	for step in range(iter_limit):
		# step 2: assign points to medoids
		pre_mediods = mediods
		sub_dis_matrix = distance_matrix[:, mediods]								# [N, K]
		min_dis, cluster_assginment = torch.min(sub_dis_matrix, dim=-1)				# [N]

		# step 3: compute medoids
		cluster_assgin_r = cluster_assginment.unsqueeze(0).repeat(K, 1)				# [K, N]
		mask = (cluster_assgin_r == K_index)										# [K, N]
		sub_matrix = repeat_dis_m * mask.unsqueeze(-1) * mask.unsqueeze(-2)
		mediods = torch.argmin(torch.sum(sub_matrix, dim=-1), dim=-1)				# [K]

		center_shift = torch.sum(torch.sqrt(torch.sum((X[mediods, :] - X[pre_mediods, :]) ** 2, dim=-1)))
		if center_shift < threshold:
			break	

	if id_sort:
		mediods, _ = torch.sort(mediods)
		# step 2: assign points to medoids
		sub_dis_matrix = distance_matrix[:, mediods]								# [N, K]
		min_dis, cluster_assginment = torch.min(sub_dis_matrix, dim=-1)				# [N]

	return cluster_assginment, mediods


if __name__ == "__main__":
	pass
