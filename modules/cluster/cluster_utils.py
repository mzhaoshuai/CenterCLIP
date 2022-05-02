# coding=utf-8
import torch
import numpy as np
from numpy.random import randint


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def pairwise_distance(data1, data2, metric='euclidean',
						self_nearest=True, all_negative=False, p=2.0):
	"""
	pairwise distance
	Args:
		data1: 	torch.Tensor, [N1, L] or [B, N1, L]
		data2: 	torch.Tensor, [N2, L] or [B, N2, L]
		metric:	(str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
		self_nearest: If ture, make sure the closest point of each point is itself
		all_negative: If True, make sure the distance values are negative 
	Return:
		a distance matrix [N1, N2] or [B, N1, N2]
	"""
	if metric == 'euclidean':
		dis = torch.cdist(data1, data2, p=p)
	
	elif metric == 'cosine':
		A_norm = data1 / (data1.norm(dim=-1, keepdim=True) + 1e-6)
		B_norm = data2 / (data2.norm(dim=-1, keepdim=True) + 1e-6)
		if data1.ndim == 3:
			dis = 1.0 - torch.bmm(A_norm, B_norm.transpose(-2, -1))
		else:
			dis = 1.0 - torch.matmul(A_norm, B_norm.transpose(-2, -1))

	else:
		raise NotImplementedError("{} metric is not implemented".format(metric))

	if all_negative:
		dis = dis - torch.max(dis) - 1.0

	if self_nearest:
		# avoid two same points
		diag = torch.arange(dis.shape[-1], device=dis.device, dtype=torch.long)
		dis[..., diag, diag] -= 1.0

	return dis


def kmeans_plusplus_init(distance_matrix, K):
	"""
	https://en.wikipedia.org/wiki/K-means%2B%2B
	In data mining, k-means++ is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm.
	The exact algorithm is as follows:
		1.Choose one center uniformly at random among the data points.
		2.For each data point x not chosen yet, compute D(x),
			the distance between x and the nearest center that has already been chosen.
		3.Choose one new data point at random as a new center,
			using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
		4.Repeat Steps 2 and 3 until k centers have been chosen.
		5.Now that the initial centers have been chosen, proceed using standard k-means clustering.
	Here, in step 3, we directly use the point x with the largest distance.
	Args:
		distance_matrix: [N, N] pairwise distance matrix 
		K: the number of clusters
	Return:
		the indices of initilized clusters
	"""
	N = distance_matrix.shape[0]
	medoids = torch.arange(K, device=distance_matrix.device, dtype=torch.long)
	medoids[0] = torch.randint(0, N, (1,))
	for i in range(1, K):
		sub_dis_matrix = distance_matrix[:, medoids[:i]]
		values, indices = torch.min(sub_dis_matrix, dim=-1)
		values_, indices_ = torch.max(values, dim=0)
		medoids[i] = indices_

	return medoids


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def KKZ_init(X, distance_matrix, K, batch=False):
	"""
	KKZ initilization for kmeans
	1. Choose the point with the maximum L2-norm as the first centroid.
	2. For j = 2, . . . ,K, each centroid μj is set in the following way: For
	any remaining data xi, we compute its distance di to the existing cen-
	troids. di is calculated as the distance between xi to its closest existing
	centroid. Then, the point with the largest di is selected as μj .
	
	Reference:
		I. Katsavounidis, C.-C. J. Kuo, and Z. Zhang. A new initialization tech-
		nique for generalized Lloyd iteration. IEEE Signal Processing Letters,
		1(10):144–146, 1994.

	"""
	l2_norm = torch.norm(X, dim=-1)
	if not batch:
		medoids = torch.arange(K, device=distance_matrix.device, dtype=torch.long)
		_, medoids[0] = torch.max(l2_norm, dim=0)
		for i in range(1, K):
			sub_dis_matrix = distance_matrix[:, medoids[:i]]
			# print(sub_dis_matrix.shape)
			values, indices = torch.min(sub_dis_matrix, dim=1)
			medoids[i] = torch.argmax(values, dim=0)
			
		# import pdb; pdb.set_trace()
		return medoids

	else:
		# batch version
		batch_i = torch.arange(X.shape[0], dtype=torch.long, device=X.device).unsqueeze(1)
		medoids = torch.arange(K, device=distance_matrix.device, dtype=torch.long)
		medoids = medoids.unsqueeze(0).repeat(X.shape[0], 1)
		_, medoids[:, 0] = torch.max(l2_norm, dim=1)
		for i in range(1, K):
			sub_dis_matrix = distance_matrix[batch_i, medoids[:, :i], :]			# [B, i, N]
			values, indices = torch.min(sub_dis_matrix, dim=1)						# [B, N]
			values_, indices_ = torch.max(values, dim=1)							# [B]
			medoids[:, i] = indices_

		return medoids


def batched_cdist_l2(x1, x2):
	"""batched pairwise ||x1 - x2||_2^2
	"""
	x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
	x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
	res = torch.baddbmm(
		x2_norm.transpose(-2, -1),
		x1,
		x2.transpose(-2, -1),
		alpha=-2
	).add(x1_norm)	#.clamp_min_(1e-30)	#.sqrt_()

	return res


def token_sparse_sampling(target, total, random_shift=True):
	"""
	randomly / uniformly sample target indices from total
	An video frame sampling example
	A video: 								1 2 3 4 5 6 7 8 9 10 11 12
	3 Segment:								1 2 3 4|5 6 7 8|9 10 11 12
	randomly select from each segment:		1 	   |      8|  10	
	-----------------------------------------
	Args:
		target: the target number of indices
		total: the total number of tokens
	Return:
		return offsets of the token (starts from 0)
	"""
	if random_shift:
		# add random perturbation to the frame selection
		average_duration = total // target
		if average_duration > 0:
			offsets = np.multiply(list(range(target)), average_duration) + \
						randint(average_duration, size=target)

		elif total > target:
			# this line suits for data_length = 1
			offsets = np.sort(np.random.choice(total, target, replace=False))
		else:
			offsets = np.clip(np.arange(0, target), 0, total)

	else:
		# general uniform sampling
		if total > target:
			tick = total / float(target)
			# center of the segment, tick / 2.0
			offsets = [int(tick / 2.0 + tick * x) for x in range(target)]
			offsets = np.array(offsets)

		else:
			offsets = np.clip(np.arange(0, target), 0, total)

	return torch.from_numpy(offsets)


if __name__ == "__main__":
	# import time
	# a = torch.randn(512, 200, 768).cuda()
	# b = torch.randn(512, 100, 768).cuda()

	# for i in range(5):
	# 	start_time = time.time()
	# 	res1 = torch.cdist(a, b)
	# 	torch.cuda.synchronize()
	# 	print(f'torch cdist time {i}: {time.time() - start_time:.3f}s')

	# for i in range(5):
	# 	start_time = time.time()
	# 	res2 = batched_cdist_l2(a, b)
	# 	torch.cuda.synchronize()
	# 	print(f'my cdist time {i}: {time.time() - start_time:.3f}s')
		
	# print(torch.sum(torch.abs(res1 - res2)))
	# print(res2.shape)

	print(token_sparse_sampling(12, 30))

