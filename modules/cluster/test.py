# coding=utf-8
"""
test functions for clustering algorithms
"""
import time
import torch

from .cluster_utils import pairwise_distance
from .kmeans import kmedoids, batch_kmedoids
from .spectral import batch_spectral_clustering
from .fast_kmeans import fast_kmedoids, batch_fast_kmedoids_with_split


def data_generate(cluster=10):
	data_l = []
	for i in range(cluster):
		data_l.append(torch.rand(4, 768) + (i + 1))
	data = torch.cat(data_l, dim=0)
	return data.unsqueeze(0).repeat(64 * 6, 1, 1)


def kmedoids_vs_fast_kmedoids_speed_test():
	"""speed test for kmedoids() and fast_kmedoids()"""
	# speed test
	iters = 200
	X = torch.rand(1000, 10)
	X = X.cuda(0)
	K = 49
	distance_matrix = pairwise_distance(X, X, metric='euclidean', all_negative=True)

	# warm up
	for i in range(10):
		N = torch.matmul(X, X.transpose(0, 1))	

	torch.cuda.synchronize()
	start = time.time()
	for i in range(10):
		assign, ids = kmedoids(X, K, distance_matrix, threshold=1e-4, iter_limit=iters, id_sort=True)
	torch.cuda.synchronize()
	end = time.time()

	print("kmedoids", end - start)
	print("max_mem", torch.cuda.max_memory_allocated())


	torch.cuda.synchronize()
	start = time.time()
	for i in range(10):
		assign_f, ids_f = fast_kmedoids(X, K, distance_matrix, threshold=1e-4, iter_limit=iters, id_sort=True)
	torch.cuda.synchronize()
	end = time.time()

	print("fast_kmedoids", end - start)
	print("max_mem", torch.cuda.max_memory_allocated())

	print("difference of assign: {}".format(torch.sum(torch.abs(assign - assign_f))))
	print("difference of ids: {}".format(torch.sum(torch.abs(ids - ids_f))))
	print("id1: {}".format(ids))
	print("id2: {}".format(ids_f))


def batch_kmedoids_speed_test():
	"""speed test for batch_kmedoids() and batch_fast_kmedoids_with_split()"""
	# speed test
	iters = 200
	K = 49
	X = data_generate(cluster=K)
	X = X.cuda(0)
	print(X.shape)
	# warm up
	for i in range(10):
		N = torch.matmul(X, X.transpose(-2, -1))	

	torch.cuda.synchronize()
	start = time.time()
	for i in range(10):
		assign, ids = batch_kmedoids(X, K, distance='euclidean', threshold=1e-5, iter_limit=60,
										id_sort=True, batch_distance=True, norm_p=1.0)
	torch.cuda.synchronize()
	end = time.time()

	print("batch kmedoids", end - start)
	print("max_mem", torch.cuda.max_memory_allocated())

	torch.cuda.synchronize()
	start = time.time()
	for i in range(10):
		# assign_f, ids_f = batch_fast_kmedoids(X, K, distance='euclidean', threshold=1e-5, iter_limit=60,
		# 										id_sort=True, norm_p=1.0)
		assign_f, ids_f = batch_fast_kmedoids_with_split(X, K, distance='euclidean', threshold=1e-5, iter_limit=60,
												id_sort=True, norm_p=1.0)

	torch.cuda.synchronize()
	end = time.time()

	print("batch fast_kmedoids", end - start)
	print("max_mem", torch.cuda.max_memory_allocated())

	torch.cuda.synchronize()
	start = time.time()
	for i in range(10):
		X.permute(1, 0, 2)
		X.permute(1, 0, 2)

	torch.cuda.synchronize()
	end = time.time()

	print("data permute time", end - start)
	print("max_mem", torch.cuda.max_memory_allocated())

	print("difference of assign: {}".format(torch.sum(torch.abs(assign - assign_f))))
	print("difference of ids: {}".format(torch.sum(torch.abs(ids - ids_f))))
	print("id1: {}".format(ids))
	print("id2: {}".format(ids_f))


def spectral_vs_fast_kmedoids_speed_test():
	"""speed test for batch_spectral_clustering() and batch_fast_kmedoids_with_split()"""
	iters = 200
	K = 49
	X = data_generate(cluster=K)
	X = X.cuda(0)
	print(X.shape)
	# warm up
	for i in range(10):
		N = torch.matmul(X, X.transpose(-2, -1))	

	torch.cuda.synchronize()
	start = time.time()
	for i in range(10):
		assign, ids = batch_spectral_clustering(X, K, mode='HeatKernel', 
													metric='euclidean',
													threshold=1e-5, iter_limit=60,
													id_sort=True, norm_p=1.0,
													correct_sign=True,
													split_size=8)
	torch.cuda.synchronize()
	end = time.time()

	print("batch_spectral_clustering", end - start)
	print("max_mem", torch.cuda.max_memory_allocated())

	torch.cuda.synchronize()
	start = time.time()
	for i in range(10):
		# assign_f, ids_f = batch_fast_kmedoids(X, K, distance='euclidean', threshold=1e-5, iter_limit=60,
		# 										id_sort=True, norm_p=1.0)
		assign_f, ids_f = batch_fast_kmedoids_with_split(X, K, distance='euclidean', threshold=1e-5, iter_limit=60,
												id_sort=True, norm_p=1.0, split_size=8)

	torch.cuda.synchronize()
	end = time.time()

	print("batch fast_kmedoids", end - start)
	print("max_mem", torch.cuda.max_memory_allocated())


	torch.cuda.synchronize()
	start = time.time()
	for i in range(10):
		X.permute(1, 0, 2)
		X.permute(1, 0, 2)

	torch.cuda.synchronize()
	end = time.time()

	print("data permute time", end - start)
	print("max_mem", torch.cuda.max_memory_allocated())

	print("difference of assign: {}".format(torch.sum(torch.abs(assign - assign_f))))
	print("difference of ids: {}".format(torch.sum(torch.abs(ids - ids_f))))
	print("id1: {}".format(ids))
	print("id2: {}".format(ids_f))


if __name__ == "__main__":
	print('-------------------------------')
	kmedoids_vs_fast_kmedoids_speed_test()
	print('-------------------------------')
	batch_kmedoids_speed_test()
	print('-------------------------------')
	spectral_vs_fast_kmedoids_speed_test()

	"""
	results on a RTX 3090

	kmedoids_vs_fast_kmedoids_speed_test()
	-------------------------------
		kmedoids 0.11393570899963379
		max_mem 12041216
		fast_kmedoids 0.05371379852294922
		max_mem 792710656
		difference of assign: 0
		difference of ids: 0

	batch_kmedoids_speed_test()
	-------------------------------
		torch.Size([384, 196, 768])
		batch kmedoids 49.58729863166809
		max_mem 792710656
		batch fast_kmedoids 4.296473503112793
		max_mem 792710656
		data permute time 9.441375732421875e-05
		max_mem 792710656
		difference of assign: 0
		difference of ids: 0

	spectral_vs_fast_kmedoids_speed_test()
	-------------------------------
		torch.Size([384, 196, 768])
		batch_spectral_clustering 6.003682851791382
		max_mem 963345408
		batch fast_kmedoids 2.3142611980438232
		max_mem 963345408
		data permute time 5.626678466796875e-05
		max_mem 963345408
		difference of assign: 0
		difference of ids: 21120
	"""
