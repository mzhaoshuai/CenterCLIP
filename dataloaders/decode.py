# coding=utf-8
"""pyav video decoder
"""
import av
import io
import os
import sys
import lmdb
import torch
import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, CenterCrop, RandomResizedCrop
# from PIL import Image
from .sampling import multi_segments_sampling, uniform_sampling
from .transforms import TensorNormalize, TensorMultiScaleCrop, GroupToTensorBCHW


class RawVideoExtractorpyAV():
	def __init__(self, centercrop=False, size=224, is_train=True, num_segments=12,
					lmdb_dataset=None):
		"""
		Args:
			lmdb_dataset: if not None, use lmdb dataset
		"""
		self.centercrop = centercrop
		self.size = size
		self.transform = self._train_transform(self.size) if is_train else self._val_transform(self.size)
		self.train = is_train
		self.num_segments = num_segments
		self.lmdb_dataset = lmdb_dataset

	def _train_transform(self, n_px):
		return Compose([
			GroupToTensorBCHW(div=True),
			# TensorMultiScaleCrop(n_px, [1, .875, .75, .66]),
			# RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
			CenterCrop(n_px),
			# TensorRandomHorizontalFlip(),
			TensorNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
			# Permute((1, 0, 2, 3))	# [T, C, H, W]
		])

	def _val_transform(self, n_px):
		return Compose([
			GroupToTensorBCHW(div=True),
			CenterCrop(n_px),
			TensorNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
			# Permute((1, 0, 2, 3))	# [T, C, H, W]
		])

	def get_video_data(self, video_path, start_time=None, end_time=None, random_shift=None):
		# pyav decode -- container and streams
		random_shift = True if random_shift is None else random_shift
		if self.lmdb_dataset in [None, 'None']:
			assert os.path.exists(video_path), "{} does not exist".format(video_path)
			container = av.open(video_path)
		else:
			key = os.path.basename(video_path)
			data = self.db_txn.get(key.encode())
			container = av.open(io.BytesIO(data))

		video_stream = container.streams.video[0]
		num_frames, fps = video_stream.frames, float(video_stream.average_rate)

		# pyav decode -- extract video frames from video stream
		sampled_frames, all_frames = [], []
		if end_time is None or start_time is None:
			for frame in container.decode(video=0):
				all_frames.append(frame)
		else:
			start_, end_ = max(0, int(start_time * fps)), min(int(end_time * fps), num_frames)
			cnt = 0
			# print(video_path, start_, end_)
			for frame in container.decode(video=0):
				if cnt >= start_ and cnt <= end_:
					all_frames.append(frame)
				cnt += 1
		# some frames may lost in the video
		num_frames = min(num_frames, len(all_frames))

		# frames sampling
		if self.train:
			inds = multi_segments_sampling(self.num_segments, num_frames, random_shift=random_shift)
		else:
			inds = uniform_sampling(self.num_segments, num_frames, twice_sample=False)

		# av.Frames --> np.ndarray of shape [H, W, C]
		try:
			sampled_frames = [all_frames[i].to_rgb().to_ndarray() for i in inds]
		except IndexError:
			print('The num_frames / fps are {} / {}, the inds are {}, '
					'the length of all frames are {}'.format(num_frames, fps, inds, len(all_frames)))
			print('The video file is {}'.format(video_path))
			sys.exit(1)

		# tensor of shape [T, C, H, W]
		video_tensor = self.transform(sampled_frames)

		frame_length = min(num_frames, self.num_segments)
		# return a video tensor, the real sampled frame for video mask
		return video_tensor, frame_length

	def process_raw_data(self, raw_video_data):
		tensor_size = raw_video_data.size()
		tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
		return tensor

	def process_frame_order(self, raw_video_data, frame_order=0):
		# 0: ordinary order; 1: reverse order; 2: random order.
		if frame_order == 0:
			pass
		elif frame_order == 1:
			reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
			reverse_order = torch.from_numpy(reverse_order).long().to(raw_video_data.device)
			raw_video_data = raw_video_data[reverse_order, ...]
		elif frame_order == 2:
			random_order = np.arange(raw_video_data.size(0))
			np.random.shuffle(random_order)
			random_order = torch.from_numpy(random_order).long().to(raw_video_data.device)
			raw_video_data = raw_video_data[random_order, ...]
		return raw_video_data

	def __getstate__(self):
		state = self.__dict__
		state["db_txn"] = None
		return state

	def __setstate__(self, state):
		# https://github.com/pytorch/vision/issues/689
		self.__dict__ = state
		if self.lmdb_dataset not in [None, 'None']:
			env = lmdb.open(self.lmdb_dataset, subdir=os.path.isdir(self.lmdb_dataset),
									readonly=True, lock=False,
									readahead=False, meminit=False,
									map_size=1<<41,)
			self.db_txn = env.begin(write=False)


if __name__ == "__main__":
	pass
