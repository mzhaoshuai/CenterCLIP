# coding=utf-8
from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import json
import torch
import logging
import numpy as np
from torch.utils.data import Dataset
from .decode import RawVideoExtractorpyAV
# from dataloaders.rawvideo_util import RawVideoExtractor


class DiDeMo_DataLoader(Dataset):
	def __init__(
			self,
			subset,
			data_path,
			features_path,
			tokenizer,
			max_words=64,
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
		# # 0: ordinary order; 1: reverse order; 2: random order.
		self.frame_order = frame_order
		assert self.frame_order in [0, 1, 2]
		# # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
		# self.slice_framepos = slice_framepos
		# assert self.slice_framepos in [0, 1, 2]

		self.subset = subset
		assert self.subset in ["train", "val", "test"]

		video_id_path_dict = {}
		video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
		video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
		video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")

		video_json_path_dict = {}
		video_json_path_dict["train"] = os.path.join(self.data_path, "train_data.json")
		video_json_path_dict["val"] = os.path.join(self.data_path, "val_data.json")
		video_json_path_dict["test"] = os.path.join(self.data_path, "test_data.json")

		with open(video_id_path_dict[self.subset], 'r') as fp:
			video_ids = [itm.strip() for itm in fp.readlines()]

		# remove missing files
		missing_files = ['37996615073@N01_3336195519_579ea4136c.3gp',
							'44124421772@N01_2867159874_e39e716b7e.mpg',
							'59627558@N00_4659075184_253744838b.3gp']
		for m in missing_files:
			if m in video_ids: video_ids.remove(m)

		caption_dict = {}
		with open(video_json_path_dict[self.subset], 'r') as f:
			json_data = json.load(f)
		for itm in json_data:
			description = itm["description"]
			times = itm["times"]
			video = itm["video"]
			if video not in video_ids:
				continue

			# each video is split into 5-second temporal chunks
			# average the points from each annotator
			start_ = np.mean([t_[0] for t_ in times]) * 5
			end_ = (np.mean([t_[1] for t_ in times]) + 1) * 5
			if video in caption_dict:
				caption_dict[video]["start"].append(start_)
				caption_dict[video]["end"].append(end_)
				caption_dict[video]["text"].append(description)
			else:
				caption_dict[video] = {}
				caption_dict[video]["start"] = [start_]
				caption_dict[video]["end"] = [end_]
				caption_dict[video]["text"] = [description]

		for k_ in caption_dict.keys():
			caption_dict[k_]["start"] = [0]
			# trick to save time on obtaining each video length
			# [https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md]:
			# Some videos are longer than 30 seconds. These videos were truncated to 30 seconds during annotation.
			caption_dict[k_]["end"] = [31]
			caption_dict[k_]["text"] = [" ".join(caption_dict[k_]["text"])]

		# get the video path
		video_dict = {}
		for root, dub_dir, video_files in os.walk(self.features_path):
			for video_file in video_files:
				video_id_ = video_file
				if video_id_ in video_ids:
					video_dict[video_id_] = os.path.join(root, video_file)					
				# our files may have name like '99996214@N00_5277035407_bb8738f945.3gp.mp4'
				if video_id_[:-4] in video_ids:
					video_dict[video_id_[:-4]] = os.path.join(root, video_file)

		self.caption_dict = caption_dict
		self.video_dict = video_dict
		# ensure no missing files
		video_ids = list(set(video_ids) & set(self.caption_dict.keys()) & set(self.video_dict.keys()))

		# get all captions
		self.iter2video_pairs_dict = {}
		for video_id in self.caption_dict.keys():
			if video_id not in video_ids:
				continue
			caption = self.caption_dict[video_id]
			n_caption = len(caption['start'])
			for sub_id in range(n_caption):
				self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (video_id, sub_id)

		# self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
		self.rawVideoExtractor = RawVideoExtractorpyAV(size=image_resolution,
														is_train=(subset == 'train'),
														num_segments=self.max_frames,
														lmdb_dataset=self.lmdb_dataset)
		self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
							  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

	def __len__(self):
		return len(self.iter2video_pairs_dict)

	def _get_text(self, video_id, sub_id):
		caption = self.caption_dict[video_id]
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
			pass
			# raise e
		# torch.Tensor of shape [Pair, T, C, H, W]
		video = torch.stack(video_list, dim=0)

		for i, v_length in enumerate(max_video_length):
			video_mask[i][:v_length] = [1] * v_length

		return video, video_mask

	def __getitem__(self, feature_idx):
		video_id, sub_id = self.iter2video_pairs_dict[feature_idx]

		pairs_text, pairs_mask, pairs_segment, starts, ends = self._get_text(video_id, sub_id)
		video, video_mask = self._get_rawvideo(video_id, starts, ends)
		return pairs_text, pairs_mask, pairs_segment, video, video_mask
