# coding=utf-8
"""
Some deep clustering methods (this method does not work now)

Reference:
[1] https://github.com/vlukiyanov/pt-dec/blob/master/ptdec/cluster.py

[2] Towards K-means-friendly Spaces: Simultaneous Deep Learning and Clustering,
https://arxiv.org/abs/1610.04794
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from .cluster_utils import batched_cdist_l2


def get_deep_cluster(width, block_id, args=None):
	"""
	Args:
		block_id: the number of the block, start from 1
	"""
	if args is None or not args.deep_cluster:
		return None

	cluster_num = args.cluster_num_blocks[block_id - 1]
	before_cluster_num = args.cluster_num_blocks[max(block_id - 2, 0)]
	after_block_frames = args.target_frames_blocks[block_id - 1]
	before_block_frames = args.target_frames_blocks[max(block_id - 2, 0)]
	frame_duration = before_block_frames // after_block_frames

	# whether clustering
	is_cluster = (cluster_num is not None and cluster_num > 1) and \
					(before_block_frames > after_block_frames or
						before_cluster_num > cluster_num)

	if not is_cluster:
		return None

	return DeepCluster(feature_dim=width,
						intermediate_dim=args.cluster_inter_dim,
						before_cluster_num=before_cluster_num,
						cluster_num=cluster_num,
						before_block_frames=before_block_frames,
						after_block_frames=after_block_frames,
						block_id=block_id,
						alpha=1.0,
						loss_type='wcss'
					)


class DeepCluster(nn.Module):
	def __init__(self, feature_dim=768,
						intermediate_dim=768,
						before_cluster_num=49,
						cluster_num=49,
						before_block_frames=12,
						after_block_frames=12,
						block_id=1,
						alpha=1.0,
						loss_type='wcss'
		):
		"""
		A deep cluster model
		Args:
			feature_dim： the dimension of input features
			before_block_frames: frames before this blocks
			after_block_frames: frames after this blocks
			block_id: the number of the block, start from 1
		"""
		super(DeepCluster, self).__init__()
		self.feature_dim = feature_dim
		self.intermediate_dim = intermediate_dim
		self.block_id = block_id
		self.alpha = alpha
		self.loss_type = loss_type
		self.cluster_num = cluster_num
		self.before_cluster_num = before_cluster_num
		self.before_block_frames = before_block_frames
		self.after_block_frames = after_block_frames
		self.frame_duration = self.before_block_frames // self.after_block_frames

		# token fc -- centroids learning
		self.token_mlp = nn.Sequential(OrderedDict([
			("fc1", nn.Linear(self.frame_duration * before_cluster_num,
								4 * self.frame_duration * before_cluster_num)),
			("ln1", nn.LayerNorm(4 * self.frame_duration * before_cluster_num)),
			("fc2", nn.Linear(4 * self.frame_duration * before_cluster_num,
									self.frame_duration * self.cluster_num)),
			("ln2", nn.LayerNorm(self.frame_duration * self.cluster_num)),
			("fc3", nn.Linear(self.frame_duration * self.cluster_num,
									self.cluster_num)),
			("ln3", nn.LayerNorm(self.cluster_num)),
		]))

		print('[model] Initializing DeepCluster model...')
		self.apply(self.init_weights)
		print('[model] Initializing DeepCluster model...[Done]')

	def forward(self, x):
		"""
		Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
		Return:
			x after clustering, clustering loss
		"""
		num_tokens, Bt, width = x.shape
		before_frame, after_frame = self.before_block_frames, self.after_block_frames
		B = Bt // before_frame

		# LND -> NLD
		x = x.permute(1, 0, 2)

		all_class_embed = x[:, 0, :].reshape(B, before_frame, 1, width)
		# T_new x [B, 1, width]. Here we simply average the class embedding from different frames
		class_embed_split = [it.mean(dim=1) for it in torch.split(all_class_embed, self.frame_duration, dim=1)]
		class_embed_tmp = torch.stack(class_embed_split, dim=1).reshape(B * after_frame, 1, width)

		res_x = x[:, 1:, :].reshape(B, before_frame, num_tokens - 1, width)
		# after_block_frames x [B, frame_duration, num_tokens - 1, width]
		frame_split = torch.split(res_x, self.frame_duration, dim=1)
		data = torch.cat(frame_split, dim=0).contiguous().reshape(B * after_frame, -1, width)

		# detach from backbone
		d_data = data.detach()

		# clustering block
		# [B * after_frame, width, frame_duration * (num_tokens - 1)]
		# [B * after_frame, (num_tokens - 1), width]
		centroids = self.token_mlp(d_data.transpose(-1, -2)).transpose(-1, -2)

		# get assign and loss
		if self.training:
			cluster_loss, assign = self._cluster_loss(d_data, centroids)
		else:
			cluster_loss = 0.0

		# [B * after_frame, (num_tokens - 1)]
		medoids = get_medoids(d_data, centroids).detach()
		# [B * after_frame, 1]
		batch_i = torch.arange(data.shape[0], dtype=torch.long, device=d_data.device).unsqueeze(1)
		new_data = data[batch_i, medoids, ...]

		# [B x after_frame, cluster, width]
		sampled_x = torch.stack(torch.split(new_data, B, dim=0), dim=1).reshape(B * after_frame, self.cluster_num, width)

		# [B x after_frame, cluster + 1, width]
		x = torch.cat([class_embed_tmp, sampled_x], dim=1).contiguous()

		# NLD -> LND
		x = x.permute(1, 0, 2)

		return x, cluster_loss

	def _cluster_loss(self, x, centroids):
		"""
		calculate the loss for clustering
		Args:
			x: [B, L, D]
			centroids: [B, K, D]
		"""
		if self.loss_type == 'wcss':
			cluster_loss, assign = batch_within_cluster_SSE(x, centroids)
		else:
			raise NotImplementedError

		return cluster_loss, assign

	def _batch_soft_assignment(self, x, centroids):
		"""
		Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
		for each cluster.
		where the Student's t-distribution is used measure similarity between feature vector and each
		cluster centroid.
		"Unsupervised Deep Embedding for Clustering Analysis" of
		Junyuan Xie, Ross Girshick, Ali Farhadi (https://arxiv.org/abs/1511.06335)
		Args:
			x: torch.Tensor [B, L, D]
			centroids: torch.Tensor [B, K, D]
		return:
			torch.Tensor [B, L, K]
		"""
		# distance matrix, [B, L, K]
		dis = torch.cdist(x, centroids, p=2.0).pow(2)
		power = (self.alpha + 1) / 2.0
		numerator = torch.pow(1.0 / (1.0 + dis / self.alpha), power)
		return numerator / torch.sum(numerator, dim=-1, keepdim=True)
	
	def init_weights(self, module):
		""" Initialize the weights for {nn.Linear, nn.Embedding, LayerNorm}.
		"""
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.01)
		elif isinstance(module, nn.LayerNorm):
			if 'beta' in dir(module) and 'gamma' in dir(module):
				module.beta.data.zero_()
				module.gamma.data.fill_(1.0)
			else:
				module.bias.data.zero_()
				module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()


def batch_within_cluster_SSE(x, centroids):
	"""
	within-cluster sum of squares (WCSS)
	Args:
		x: [B, L, D]
		centroids: [B, K, D]
	"""
	B, K = centroids.shape[0], centroids.shape[1]
	# [B, L, K], distance matrix
	# distance_matrix = torch.cdist(x, centroids, p=2.0).pow(2.0)
	distance_matrix = batched_cdist_l2(x, centroids)
	# [B, L]
	values, indices = torch.min(distance_matrix, dim=-1)
	wcss = torch.sum(values, dim=-1).mean()

	return wcss, indices


def get_medoids(x, centroids):
	"""
	get the indices of medoids according to centroids
	Args:
		x: [B, L, D]
		centroids: [B, K, D]
	"""
	B, K = centroids.shape[0], centroids.shape[1]
	# [B, K, 1]
	K_index = torch.arange(K, dtype=torch.long, device=x.device).reshape(1, K, 1).repeat(B, 1, 1)	

	# [B, L, K], distance matrix, all negative
	distance_matrix = torch.cdist(x, centroids, p=2.0)
	distance_matrix = distance_matrix - torch.max(distance_matrix) - 1.0

	# [B, L]
	cluster_distance, assign = torch.min(distance_matrix, dim=-1)

	rep_distance_matrix = cluster_distance.unsqueeze(1).repeat(1, K, 1)			# [B, K, L]
	rep_assign = assign.unsqueeze(1).repeat(1, K, 1)							# [B, K, L]
	mask = (rep_assign == K_index)												# [B, K, L]
	# [B, K]
	values, indices = torch.min(mask * rep_distance_matrix, dim=-1)

	mediods, _ = torch.sort(indices, dim=1)
	return mediods


class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)
