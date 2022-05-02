# coding=utf-8
"""
Token Cluster class
"""
import os
import torch
import logging
import numpy as np
from .spectral import batch_spectral_clustering, spatial_temporal_graph
from .cluster_utils import token_sparse_sampling
from .fast_kmeans import batch_fast_kmedoids_with_split
from .shift import token_shift, temporal_shift_wo_cls


def get_cluster_inter(width, block_id, args=None):
	"""
	Args:
		block_id: the number of the block, start from 1
	"""
	if args is None or not args.cluster_inter:
		return None

	target_frames_blocks = [args.max_frames,] + args.target_frames_blocks

	cluster_num = args.cluster_num_blocks[block_id - 1]
	before_cluster_num = args.cluster_num_blocks[max(block_id - 2, 0)]
	after_block_frames = target_frames_blocks[block_id]
	before_block_frames = target_frames_blocks[block_id - 1]
	# frame_duration = before_block_frames // after_block_frames

	# whether clustering
	is_cluster = (cluster_num is not None and cluster_num > 1) and \
					(before_block_frames > after_block_frames or
						before_cluster_num > cluster_num)

	if not is_cluster:
		return None

	return TokenClusterInter(algorithm=args.cluster_algo,
								block_id=block_id,
								before_cluster_num=before_cluster_num,
								cluster_num=cluster_num,
								before_block_frames=before_block_frames,
								after_block_frames=after_block_frames,
								original_frame=args.max_frames,								
								distance=args.cluster_distance,
								threshold=args.cluster_threshold,
								iter_limit=args.cluster_iter_limit,
								id_sort=True,
								norm_p=args.minkowski_norm_p,
								spectral_sigma=args.spectral_sigma,
								spectral_graph=args.spectral_graph,
								spectral_knn_k=args.spectral_knn_k,
								spectral_spatial_temporal_graph=args.spectral_spg,
								aggregation=args.aggregation,
								split_size=4 if args.pretrained_clip_name == 'ViT-B/16' else 16,
								cluster_embedding=args.cluster_embedding,
								cluster_frame_embedding=args.cluster_frame_embedding,
								adaptive_cls=False,
								save_feature_path=args.save_feature_path,
								svd_correct_sign=args.svd_correct_sign,
								pre_norm=args.pre_norm
							)


class TokenClusterInter(torch.nn.Module):
	def __init__(self, algorithm='kmediods++',
					block_id=1,
					before_cluster_num=49,
					cluster_num=49,
					before_block_frames=12,
					after_block_frames=12,
					original_frame=12,
					distance='euclidean',
					threshold=1e-6,
					iter_limit=80,
					id_sort=True,
					aggregation=None,
					split_size=8,
					norm_p=2.0,
					spectral_graph='HeatKernel',
					spectral_sigma=2.0,
					spectral_knn_k=0,
					spectral_spatial_temporal_graph=False,
					cluster_embedding=False,
					cluster_frame_embedding=False,
					adaptive_cls=False,
					mean_residual=False,
					transformer_width=768,
					save_feature_path=None,
					svd_correct_sign=1,
					pre_norm=False):
		"""
		Add TokenCluster in the blocks of the transformers
		Args:
			algorithm:  the cluster algorithm used
			before_cluster_num: the number of tokens per frame before this module
			cluster_num: the number of tokens per temporal segment after this module
			before_block_frames: the number of frames before this module
			after_block_frames: the number of frames after this module
			original_frame: the number of original input frames
			distance: distance metric used in clustering algorithm, [options: 'euclidean', 'cosine']
			threshold: stop threshold for clustering
			iter_limit: max iteration steps for clustering
			id_sort: whether sort id of cluster centers in ascending order 
			aggregation: how to aggregate the cluster centers, 'None' for only use cluster centers and abondan non-center
							tokens; other options will use the mean of tokens within the cluster 
			split_size: applying data spliting to avoid OOM of GPU mem
			norm_p: norm of distance metric
			spectral_graph: grap choices of spectral clustering, [options: 'HeatKernel', 'KNN']
			spectral_sigma: sigma / variance of gaussian function in spectral clustering
			spectral_knn_k: K for 'KNN' graph in spectral clustering
			spectral_spatial_temporal_graph: only reserve affinity within certain spatial or temporal distance,
							see the function for details
			cluster_embedding: cluster embedding for different cluster centers
			cluster_frame_embedding: add frame embedding for frames in a temporal segment
			adaptive_cls: apply learnable mutiplier for [CLASS] embedding aggregation
			mean_residual: use the mean of frames as residual connection
			save_feature_path: path to save intermediate features or clsuter center ids
			svd_correct_sign: resolve the sign ambiguity of SVD or not
			pre_norm: if true, do l2 normalization first before clustering
		"""
		super().__init__()
		assert algorithm in ['kmediods++', 'pooling', 'sparse_sampling', 'spectral',
								'temporal_shift', 'token_shift']
		self.save_feature_path = save_feature_path
		self.algorithm = algorithm
		self.original_frame = original_frame
		self.before_cluster_num = before_cluster_num
		self.cluster_num = cluster_num
		self.before_block_frames = before_block_frames
		self.after_block_frames = after_block_frames
		self.frame_duration = before_block_frames // after_block_frames
		self.distance = distance
		self.threshold = threshold
		self.iter_limit = iter_limit
		self.id_sort = id_sort
		self.aggregation = aggregation
		self.split_size = split_size
		self.norm_p = norm_p
		self.mean_residual = mean_residual
		self.spectral_graph = spectral_graph
		self.spectral_sigma = spectral_sigma
		self.pre_norm = pre_norm
		# when K of spectral_knn_k is small, use an adaptive number
		if spectral_knn_k < 5:
			self.spectral_knn_k = int(5 * self.frame_duration) if before_cluster_num < 100 \
									else int(5 * self.frame_duration + 5)
		else:
			self.spectral_knn_k = spectral_knn_k
		self.spectral_spatial_temporal_graph = spectral_spatial_temporal_graph
		self.svd_correct_sign = svd_correct_sign

		self.cluster_embedding = cluster_embedding if algorithm in ['kmediods++', 'spectral'] else False
		self.cluster_frame_embedding = cluster_frame_embedding if algorithm in ['kmediods++', 'spectral'] else False
		self.adaptive_cls = adaptive_cls if algorithm in ['kmediods++', 'spectral'] else False
		self.shift_fold_div = 8

		# create some new parameters for cluster
		scale = transformer_width ** -0.5
		if self.cluster_embedding:
			# including [CLASS]
			# self.cluster_cls_embed = torch.nn.Parameter(scale * torch.randn(1, transformer_width))
			self.cluster_embed = torch.nn.Parameter(scale * torch.randn(self.cluster_num, transformer_width))
			# no [CLASS]
			# self.cluster_embed = torch.nn.Parameter(scale * torch.randn(self.cluster_num, transformer_width))
		if self.cluster_frame_embedding:
			self.cluster_frame_embed = torch.nn.Parameter(scale *
											torch.randn(self.frame_duration, transformer_width).unsqueeze(1))			
		if self.adaptive_cls:
			m = [1 / self.frame_duration for i in range(self.before_block_frames)]
			self.cls_multiplier = torch.nn.Parameter(torch.tensor(m).float().reshape(1, self.before_block_frames, 1, 1))

		if self.spectral_spatial_temporal_graph:
			s_kernel = 9 if before_cluster_num < 100 else 19
			t_kernel = 7
			spg = spatial_temporal_graph(before_cluster_num * self.frame_duration,
											before_cluster_num, s_kernel=s_kernel, t_kernel=t_kernel)
			# shape [1, before_cluster_num * frame_duration, before_cluster_num * frame_duration]
			self.register_buffer("spg", spg.unsqueeze(0).float())
		else:
			self.spg = None
		self.cnt = 1

		if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
			logging.info("[cluster] Creating cluster algorithm object in transformer blocks:\n"
							"\t algorithm: {}\n"
							"\t block_id (start 1): {}\n"
							"\t cluster: {} --> {}\n"
							"\t frames: {} --> {}\n"
							"\t cluster_embedding / cluster_frame_embedding / adaptive_cls: [{} / {} / {}]\n"
							"\t split_size: {}\n"
							"\t distance / pre_norm / distance norm_p: [{} / {} / {}]\n"
							"\t stop threshold / iter_limit: [{} / {}]\n"
							"\t spectral_graph / sigma / knn_k / spg / sign correct: [{} / {} / {} / {} / {}]\n"
							"\t mean_residual: {}".format(algorithm, block_id,
							before_cluster_num, cluster_num,
							before_block_frames, after_block_frames,
							cluster_embedding, cluster_frame_embedding, adaptive_cls,
							split_size, distance, self.pre_norm,
							norm_p, threshold, iter_limit,
							spectral_graph, spectral_sigma, self.spectral_knn_k,
							spectral_spatial_temporal_graph, svd_correct_sign, mean_residual))
			logging.info('[cluster] Creating cluster algorithm object in transformer blocks...[DONE]\n')

	def forward(self, x):
		"""
		Args:
			x: torch.Tensor of shape [grid x grid + 1, B x T, width] / [L, N, D]
			block_id: the number of the block, start from 1
		"""
		num_tokens, Bt, width = x.shape
		before_block_frames, after_block_frames = self.before_block_frames, self.after_block_frames
		before_cluster_num, cluster_num = self.before_cluster_num, self.cluster_num
		frame_duration = self.frame_duration
		B = Bt // before_block_frames

		# uncomment of you want to save intermediate features of CLIP
		# if self.save_feature_path is not None and os.path.exists(self.save_feature_path):
		# 	if self.cnt <= 2000:
		# 		feature = x.permute(1, 0, 2).reshape(B, before_block_frames, -1, width)
		# 		np_arr = feature.cpu().detach().numpy()
		# 		with open(os.path.join(self.save_feature_path, 'feature_' + str(self.cnt) + '.npy'), 'wb') as f:
		# 			np.save(f, np_arr)
		# 		self.cnt += 1
		# 	return x, None

		if self.mean_residual:
			assert num_tokens == (cluster_num + 1)
			residual_x = x.reshape(num_tokens, B, before_block_frames, width)
			# after_block_frames x [num_tokens, B, frame_duration, width]
			frame_split = [it.mean(dim=2) for it in torch.split(residual_x, frame_duration, dim=2)]
			# [num_tokens, B x target_frames, width]
			residual_x = torch.stack(frame_split, dim=2).contiguous().reshape(num_tokens, B * after_block_frames, width)
		else:
			residual_x = None

		# LND -> NLD
		x = x.permute(1, 0, 2)

		# cluster
		if self.algorithm in ['kmediods++', 'spectral']:
			all_class_embed = x[:, 0, :].reshape(B, before_block_frames, 1, width)
			if self.adaptive_cls:
				all_class_embed = all_class_embed * self.cls_multiplier

			res_x = x[:, 1:, :].reshape(B, before_block_frames, num_tokens - 1, width)
			# after_block_frames x [B, frame_duration, num_tokens - 1, width]
			frame_split = torch.split(res_x, frame_duration, dim=1)
			res_tmp = torch.cat(frame_split, dim=0).contiguous().reshape(B * after_block_frames, -1, width)
			batch_index = torch.arange(res_tmp.shape[0], dtype=torch.long, device=x.device).unsqueeze(-1)

			if self.algorithm == 'kmediods++':
				assign, mediods_ids = batch_fast_kmedoids_with_split(res_tmp, cluster_num,
														distance=self.distance, threshold=self.threshold,
														iter_limit=self.iter_limit,
														id_sort=self.id_sort, 
														norm_p=self.norm_p,
														split_size=self.split_size,
														pre_norm=self.pre_norm)
			elif self.algorithm == 'spectral':
				assign, mediods_ids = batch_spectral_clustering(res_tmp, cluster_num,
														mode=self.spectral_graph,
														knn_k=self.spectral_knn_k,
														metric=self.distance,
														threshold=self.threshold, iter_limit=self.iter_limit,
														id_sort=self.id_sort, norm_p=self.norm_p,
														correct_sign=self.svd_correct_sign,
														split_size=self.split_size,
														sigma=self.spectral_sigma,
														spatial_temporal_graph=self.spg)
			else:
				raise NotImplementedError
			# uncomment of you want to save intermediate ids of centroids
			# save id of centroids
			# if self.save_feature_path is not None and os.path.exists(self.save_feature_path):
			# 	if self.cnt <= 30:
			# 		np_arr = mediods_ids.cpu().detach().numpy()
			# 		with open(os.path.join(self.save_feature_path, 'ids_' + str(self.cnt) + '.npy'), 'wb') as f:
			# 			np.save(f, np_arr)
			# 		self.cnt += 1

			# if self.cluster_frame_embedding:
			# 	res_tmp = (res_tmp.reshape(B * after_block_frames, frame_duration, -1, width) + 
			# 				self.cluster_frame_embed.to(res_tmp.dtype)).reshape(B * after_block_frames, -1, width)

			if self.aggregation in [None, 'None']:
				# [B x T_new, cluster, width]
				x_tmp = res_tmp[batch_index, mediods_ids, ...]
			else:
				res_x_list = []
				for i in range(cluster_num):
					# [B, cluster, 1]
					mask = (assign == i).unsqueeze(-1)
					# [B, 1, width]
					x_tmp_tmp = torch.sum(res_tmp * mask, dim=1, keepdim=True) / torch.sum(
								mask.float(), dim=1, keepdim=True)
					res_x_list.append(x_tmp_tmp)
				# [B x T_new, cluster, width]
				x_tmp = torch.cat(res_x_list, dim=1)

			# [B x target_frames, cluster, width]
			x_tmp = torch.stack(torch.split(x_tmp, B, dim=0), dim=1).reshape(B * after_block_frames, cluster_num, width)
			if self.cluster_embedding:
				x_tmp = x_tmp + self.cluster_embed.to(x_tmp.dtype)	
			# T_new x [B, 1, width]. Here we simply average the class embedding from different frames
			class_embed_split = [it.mean(dim=1) for it in torch.split(all_class_embed, frame_duration, dim=1)]
			class_embed_tmp = torch.stack(class_embed_split, dim=1).reshape(B * after_block_frames, 1, width)

			x = torch.cat([class_embed_tmp, x_tmp], dim=1).contiguous()
			# if self.cluster_embedding:
				# x[:, :1, :] = x[:, :1, :] + self.cluster_cls_embed.to(x.dtype)
				# x = x + self.cluster_embed.to(x.dtype)

		elif self.algorithm == 'pooling':
			res_x = x.reshape(B, before_block_frames, num_tokens, width)
			# after_block_frames x [B, frame_duration, num_tokens, width]
			frame_split = [it.mean(dim=1) for it in torch.split(res_x, frame_duration, dim=1)]
			# [B x target_frames, num_tokens, width]
			x = torch.stack(frame_split, dim=1).contiguous().reshape(B * after_block_frames, num_tokens, width)

		elif self.algorithm == 'sparse_sampling':
			# T_new x [B, 1, width]. Here we simply average the class embedding from different frames
			all_class_embed = x[:, 0, :].reshape(B, before_block_frames, 1, width)
			class_embed_split = [it.mean(dim=1) for it in torch.split(all_class_embed, frame_duration, dim=1)]
			class_embed_tmp = torch.stack(class_embed_split, dim=1).reshape(B * after_block_frames, 1, width)

			# after_block_frames x [B, frame_duration, num_tokens - 1, width]
			res_x = x[:, 1:, :].reshape(B, before_block_frames, num_tokens - 1, width)
			frame_split = torch.split(res_x, frame_duration, dim=1)

			res_all = []
			for it in frame_split:
				it_tmp = it.reshape(B, -1, width)
				ind = token_sparse_sampling(cluster_num, it_tmp.shape[1], self.training)
				ind = ind.long().to(it.device)
				res_all.append(it_tmp[:, ind, :])

			# [B x target_frames, cluster, width]
			x_tmp = torch.stack(res_all, dim=1).contiguous().reshape(B * after_block_frames, cluster_num, width)
			x = torch.cat([class_embed_tmp, x_tmp], dim=1).contiguous()		

		elif self.algorithm == 'temporal_shift':
			x = temporal_shift_wo_cls(x, self.original_frame, fold_div=self.shift_fold_div)

		elif self.algorithm == 'token_shift':
			x = token_shift(x, self.original_frame, fold_div=self.shift_fold_div)

		# NLD -> LND
		x = x.permute(1, 0, 2)

		return x, residual_x
