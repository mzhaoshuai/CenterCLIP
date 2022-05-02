# coding=utf-8
"""
spectral clustering in PyTorch

Reference:

[1] https://github.com/tczhangzhi/cluster/blob/master/torchcluster/zoo/spectrum.py
"""
import torch
import numpy as np
from .cluster_utils import batched_cdist_l2
from .fast_kmeans import batch_fast_kmedoids, batch_fast_kmedoids_with_split


@torch.no_grad()
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def batch_spectral_clustering(X, K, mode='HeatKernel', 
								knn_k=10,
								metric='euclidean',
								threshold=1e-5, iter_limit=60,
								id_sort=True, norm_p=1.0,
								correct_sign=False,
								split_size=8,
								sigma=2.5,
								spatial_temporal_graph=None):
	"""
	perform normalized spectral clustering
	Args:
		X: (torch.tensor) matrix, dtype should be torch.float
		K: (int) number of clusters
		mode: how to construct graph for spectral clustering
		threshold: (float) threshold [default: 0.0001]
		iter_limit: hard limit for max number of iterations
		norm_p: the norm of distance metric
		correct_sign: correct sign for SVD or not
		split_size: split a batch tensor into multiple chunks in order to avoid OOM
	Return:
		(cluster_assginment, mediods)
	"""
	assert metric in ['euclidean', 'cosine'] and X.ndim == 3
	B, N, L = X.shape[0], X.shape[1], X.shape[2]
	# adjacent matrix, [B, N, N]
	W = constructW(X, X, sigma=sigma, mode=mode, knn_k=knn_k, spatial_temporal_graph=spatial_temporal_graph)											
	
	# degree matrix, [B, N, N]
	diag_D = W.sum(dim=-1)
	D = torch.diag_embed(diag_D, dim1=-2, dim2=-1)
	inv_D = torch.diag_embed(torch.pow(diag_D, -0.5))
	
	# Laplacian matrix
	L = D - W
	L_sym = torch.bmm(torch.bmm(inv_D, L), inv_D)
	
	# eigen decomposition, eigenvalues in descending oerder
	U, S, Vh = torch.linalg.svd(L_sym, full_matrices=False)
	if correct_sign:
		U = batch_sign_flip_rasmus_bro(U, S, Vh, backend="pytorch")

	# corresponding eigenvectors of K smallest eigenvalues / lower dimension representation
	Q = U[:, :, -K:]
	Q = Q / (Q.norm(p=2, dim=-1, keepdim=True) + 1e-6)

	if split_size > 1 and B > split_size:
		cluster_assginment, mediods = batch_fast_kmedoids_with_split(Q, K, distance=metric,
															threshold=threshold, iter_limit=iter_limit,
															id_sort=id_sort, norm_p=norm_p,
															split_size=split_size)
	else:
		cluster_assginment, mediods = batch_fast_kmedoids(Q, K, distance=metric,
															threshold=threshold, iter_limit=iter_limit,
															id_sort=id_sort, norm_p=norm_p)

	return cluster_assginment, mediods


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def constructW(x, y, sigma=2.0, mode='HeatKernel', knn_k=10, mutual=False, spatial_temporal_graph=None):
	"""
	create graph for spetral clustering
	Args:
		x: torch.Tensor, [B, N, D]
		y: torch.Tensor, [B, M, D]
	"""
	if mode == 'HeatKernel':
		W = torch.exp(-1.0 * batched_cdist_l2(x, y) / (2 * sigma ** 2))

	elif mode == 'KNN':
		W = torch.exp(-1.0 * batched_cdist_l2(x, y) / (2 * sigma ** 2))
		value, indices = torch.topk(W, knn_k, dim=-1, largest=True)
		k_value = value[:, :, -1:]
		mask_last = (W >= k_value)
		if mutual:
			mask = torch.logical_and(mask_last, mask_last.transpose(-2, -1))
		else:
			mask = torch.logical_or(mask_last, mask_last.transpose(-2, -1))

		W = W * mask
		# W = mask.float()

	else:
		raise NotImplementedError

	if spatial_temporal_graph is not None:
		W = W * spatial_temporal_graph

	return W


def batch_sign_flip_rasmus_bro(U, S, VT, backend="pytorch"):
	"""
	sign flip methods deseribed in
	Rasmus Bro (2021). Sign correction in SVD and PCA
	(https://www.mathworks.com/matlabcentral/fileexchange/22118-sign-correction-in-svd-and-pca),
	MATLAB Central File Exchange. Retrieved July 21, 2021.
	Make the left/right singular vectors to have the similar direction as the data points.
	
	Args:
		U, S, VT: X = U @ S @ VT, components after SVD.
		shape: X - [B, M, N], U - [B, M, M], S - [B, M/N], VT - [B, N, N].
		In our case, M = N and X =  XT.
	Return:
		the sign flipped U (left singular vector)
	"""
	if backend == "pytorch":
		# singular values time each row
		SVT = S.unsqueeze(-1) * VT
		# sum along row
		sign_left = torch.sum(torch.sign(SVT) * torch.square(SVT), dim=2)
		# times each column of U
		U = torch.sign(sign_left).unsqueeze(1) * U
	else:
		# non-batched for numpy
		SVT = np.expand_dims(S, axis=-1) * VT
		sign_left = np.sum(np.sign(SVT) * np.square(SVT), axis=-1)
		U = np.expand_dims(np.sign(sign_left), axis=0) * U

	return U


def spatial_temporal_graph(N, tokens_per_frame,s_kernel=5, t_kernel=5):
	"""construct a spatial-temporal kernel
	Args:
		N: the number of total tokens
		T: the number of video frame
		tokens_per_frame: the number of tokens per frame
	"""
	H, W = int(tokens_per_frame ** 0.5), int(tokens_per_frame ** 0.5)
	frames = N // tokens_per_frame
	graph = torch.zeros(N, N)
	half_t, half_s = t_kernel // 2, s_kernel // 2

	for i in range(N):
		t_ = i // tokens_per_frame			# start from 0
		h_ = i % tokens_per_frame // W		# start from 0
		w_ = i % tokens_per_frame % W
		connect_t = [t_ + m for m in range(-half_t, half_t + 1) if 0 <= (t_ + m) < frames ]
		connect_x = [w_ + m for m in range(-half_s, half_s + 1) if 0 <= (w_ + m) < W]
		connect_y = [h_ + m for m in range(-half_s, half_s + 1) if 0 <= (h_ + m) < H]
 
		for t  in connect_t:
			for y in connect_y:
				for x in connect_x:
					r_i = t * tokens_per_frame + y * W + x
					graph[i, r_i] = 1
	graph = graph > 0

	return graph


if __name__ == "__main__":
	pass
