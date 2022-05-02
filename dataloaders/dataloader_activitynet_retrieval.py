# coding=utf-8
from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import json
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from .decode import RawVideoExtractorpyAV
# from dataloaders.rawvideo_util import RawVideoExtractor


class ActivityNet_DataLoader(Dataset):
	def __init__(
			self,
			subset,
			data_path,
			features_path,
			tokenizer,
			max_words=30,
			feature_framerate=1.0,
			max_frames=100,
			image_resolution=224,
			frame_order=0,
			slice_framepos=0,
			lmdb_dataset=None
	):
		self.data_path = data_path
		self.features_path = features_path
		self.feature_framerate = feature_framerate
		self.max_words = max_words
		self.max_frames = max_frames
		self.tokenizer = tokenizer
		self.lmdb_dataset = lmdb_dataset

		# 0: ordinary order; 1: reverse order; 2: random order.
		self.frame_order = frame_order
		assert self.frame_order in [0, 1, 2]
		# 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
		self.slice_framepos = slice_framepos
		assert self.slice_framepos in [0, 1, 2]

		self.subset = subset
		assert self.subset in ["train", "val"]
		# get video ids
		video_id_path_dict = {}
		video_id_path_dict["train"] = os.path.join(self.data_path, "train_ids.json")
		video_id_path_dict["val"] = os.path.join(self.data_path, "val_ids.json")
		# get content
		video_json_path_dict = {}
		video_json_path_dict["train"] = os.path.join(self.data_path, "train.json")
		video_json_path_dict["val"] = os.path.join(self.data_path, "val_1.json")

		pseudo_video_id_list, video_id_list = self._get_video_id_single(video_id_path_dict[self.subset])
		pseudo_caption_dict = self._get_captions_single(video_json_path_dict[self.subset])

		print("video id list: {}".format(len(video_id_list)))
		print("pseudo caption dict: {}".format(len(pseudo_caption_dict.keys())))

		# some videos are broken
		broken_videos = ['NHznDFD3V3k', 'v_NHznDFD3V3k']
		# broken_videos = []
		for m in broken_videos:
			if m in video_id_list: video_id_list.remove(m)
			if m in pseudo_video_id_list: pseudo_video_id_list.remove(m)
	
		with open(os.path.join(self.data_path, "video_path.json"), 'r') as f:
			all_video_dict = json.load(f)
		video_dict = dict([(k, os.path.join(self.features_path, all_video_dict[k]))
								for k in video_id_list if k in all_video_dict])
		# video_dict = {}
		# for root, dub_dir, video_files in os.walk(self.features_path):
		# 	for video_file in video_files:
		# 		video_id_ = ".".join(video_file.split(".")[:-1])
		# 		if video_id_ not in video_id_list:
		# 			continue
		# 		file_path_ = os.path.join(root, video_file)
		# 		video_dict[video_id_] = file_path_
		self.video_dict = video_dict
		print("video dict: {}".format(len(video_dict)))

		self.pseudo_video_id_list = pseudo_video_id_list
		self.video_id_list = video_id_list
		self.pseudo_caption_dict = pseudo_caption_dict

		# get iterator video ids
		self.video_id2idx_dict = {pseudo_video_id: id for id, pseudo_video_id in enumerate(self.pseudo_video_id_list)}
		# Get all captions
		self.iter2video_pairs_dict = {}
		for pseudo_video_id, video_id in zip(self.pseudo_video_id_list, self.video_id_list):
			if pseudo_video_id not in self.pseudo_caption_dict or video_id not in self.video_dict:
				continue
			caption = self.pseudo_caption_dict[pseudo_video_id]
			n_caption = len(caption['start'])
			for sub_id in range(n_caption):
				self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (pseudo_video_id, sub_id)

		# self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
		self.rawVideoExtractor = RawVideoExtractorpyAV(size=image_resolution,
														is_train=(subset == 'train'),
														num_segments=self.max_frames,
														lmdb_dataset=self.lmdb_dataset)
		self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
							  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

	def __len__(self):
		return len(self.iter2video_pairs_dict)

	def _get_video_id_from_pseduo(self, pseudo_video_id):
		# pseduo_id like 'v_nlkmPF8TBdQ'
		video_id = pseudo_video_id[2:]
		return video_id

	def _get_video_id_single(self, path):
		pseudo_video_id_list = []
		video_id_list = []
		print('Loading json: {}'.format(path))
		with open(path, 'r') as f:
			json_data = json.load(f)

		for pseudo_video_id in json_data:
			if pseudo_video_id in pseudo_video_id_list:
				print("reduplicate.")
			else:
				video_id = self._get_video_id_from_pseduo(pseudo_video_id)
				pseudo_video_id_list.append(pseudo_video_id)
				video_id_list.append(video_id)
		return pseudo_video_id_list, video_id_list

	def _get_captions_single(self, path):
		pseudo_caption_dict = {}
		with open(path, 'r') as f:
			json_data = json.load(f)

		for pseudo_video_id, v_ in json_data.items():
			pseudo_caption_dict[pseudo_video_id] = {}
			duration = v_["duration"]
			pseudo_caption_dict[pseudo_video_id]["start"] = np.array([0], dtype=object)
			pseudo_caption_dict[pseudo_video_id]["end"] = np.array([int(math.ceil(float(duration)))], dtype=object)
			pseudo_caption_dict[pseudo_video_id]["text"] = np.array([" ".join(v_["sentences"])], dtype=object)
		return pseudo_caption_dict

	def _get_text(self, pseudo_video_id, sub_id):
		caption = self.pseudo_caption_dict[pseudo_video_id]
		k = 1
		r_ind = [sub_id]

		starts = np.zeros(k, dtype=np.long)
		ends = np.zeros(k, dtype=np.long)
		pairs_text = np.zeros((k, self.max_words), dtype=np.long)
		pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
		pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

		for i in range(k):
			ind = r_ind[i]
			start_, end_ = caption['start'][ind], caption['end'][ind]
			words = self.tokenizer.tokenize(caption['text'][ind])
			starts[i], ends[i] = start_, end_

			words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
			total_length_with_CLS = self.max_words - 1
			if len(words) > total_length_with_CLS:
				words = words[:total_length_with_CLS]
			words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

			input_ids = self.tokenizer.convert_tokens_to_ids(words)
			input_mask = [1] * len(input_ids)
			segment_ids = [0] * len(input_ids)
			while len(input_ids) < self.max_words:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)
			assert len(input_ids) == self.max_words
			assert len(input_mask) == self.max_words
			assert len(segment_ids) == self.max_words

			pairs_text[i] = np.array(input_ids)
			pairs_mask[i] = np.array(input_mask)
			pairs_segment[i] = np.array(segment_ids)

		return pairs_text, pairs_mask, pairs_segment, starts, ends

	def _get_rawvideo(self, idx, s, e):
		video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
		max_video_length = [0] * len(s)

		try:
			video_list = []
			video_path = self.video_dict[idx]

			for i in range(len(s)):
				start_time = int(s[i])
				end_time = int(e[i])
				start_time = start_time if start_time >= 0. else 0.
				end_time = end_time if end_time >= 0. else 0.
				if start_time > end_time:
					start_time, end_time = end_time, start_time
				elif start_time == end_time:
					end_time = end_time + 1

				# torch.Tensor of shape [T, C, H, W]
				# Well, here set random_shift=True will lost many frames when use a large value of max_frames
				raw_video_data, slice_len = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time,
																					random_shift=True)
				max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
				# raw_video_data = self.rawVideoExtractor.process_frame_order(raw_video_data, frame_order=self.frame_order)
				video_list.append(raw_video_data)

		except Exception as excep:
			print("video path: {} error. video id: {}, start: {}, end: {}, Error: {}".format(video_path, idx, s, e, excep))
			raise excep
		# torch.Tensor of shape [Pair, T, C, H, W]
		video = torch.stack(video_list, dim=0)

		for i, v_length in enumerate(max_video_length):
			video_mask[i][:v_length] = [1] * v_length

		return video, video_mask

	def __getitem__(self, feature_idx):
		pseudo_video_id, sub_id = self.iter2video_pairs_dict[feature_idx]
		idx = self.video_id2idx_dict[pseudo_video_id]

		pairs_text, pairs_mask, pairs_segment, starts, ends = self._get_text(pseudo_video_id, sub_id)
		video, video_mask = self._get_rawvideo(self.video_id_list[idx], starts, ends)
		return pairs_text, pairs_mask, pairs_segment, video, video_mask
