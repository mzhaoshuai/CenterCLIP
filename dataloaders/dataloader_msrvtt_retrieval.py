# coding=utf-8
"""dataset for MSRVTT
"""
from __future__ import absolute_import, division, unicode_literals

import os
import json
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from .decode import RawVideoExtractorpyAV


class MSRVTT_DataLoader(Dataset):
	"""MSRVTT dataset loader."""
	def __init__(
			self,
			csv_path,
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
		self.data = pd.read_csv(csv_path)
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

		self.rawVideoExtractor = RawVideoExtractorpyAV(size=image_resolution, is_train=False,
														num_segments=self.max_frames,
														lmdb_dataset=self.lmdb_dataset)
		self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
							  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

	def __len__(self):
		return len(self.data)

	def _get_text(self, video_id, sentence):
		choice_video_ids = [video_id]
		n_caption = len(choice_video_ids)

		k = n_caption
		pairs_text = np.zeros((k, self.max_words), dtype=np.long)
		pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
		pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

		for i, video_id in enumerate(choice_video_ids):
			words = self.tokenizer.tokenize(sentence)

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

		return pairs_text, pairs_mask, pairs_segment, choice_video_ids

	def _get_rawvideo(self, choice_video_ids):
		video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
		max_video_length = [0] * len(choice_video_ids)
		video_list = []
		# video data
		for i, video_id in enumerate(choice_video_ids):
			# torch.Tensor of shape [T, C, H, W]
			raw_video_data, slice_len = self.rawVideoExtractor.get_video_data(
										os.path.join(self.features_path, "{}.mp4".format(video_id))
										)
			# slice_len = raw_video_data.size(0)
			max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
			video_list.append(raw_video_data)
		# video mask
		for i, v_length in enumerate(max_video_length):
			video_mask[i][:v_length] = [1] * v_length
		# torch.Tensor of shape [Pair, T, C, H, W]
		video = torch.stack(video_list, dim=0)

		return video, video_mask

	def __getitem__(self, idx):
		video_id = self.data['video_id'].values[idx]
		sentence = self.data['sentence'].values[idx]

		pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
		video, video_mask = self._get_rawvideo(choice_video_ids)
		return pairs_text, pairs_mask, pairs_segment, video, video_mask


class MSRVTT_TrainDataLoader(Dataset):
	"""MSRVTT train dataset loader."""
	def __init__(
			self,
			csv_path,
			json_path,
			features_path,
			tokenizer,
			max_words=30,
			feature_framerate=1.0,
			max_frames=100,
			unfold_sentences=False,
			image_resolution=224,
			frame_order=0,
			slice_framepos=0,
			lmdb_dataset=None
	):
		"""
		MSRVTT training dataset.
		MSRVTT has 200000 sentences, 10000 videos.
		=============================
		Args:
			csv_path: path to the video list
			json_path: text corpus
			features_path: video path
			tokenizer: text tokenizer
			feature_framerate: fps
		"""
		self.csv = pd.read_csv(csv_path)
		self.data = json.load(open(json_path, 'r'))
		self.features_path = features_path
		self.feature_framerate = feature_framerate
		self.max_words = max_words
		self.max_frames = max_frames
		self.tokenizer = tokenizer
		self.lmdb_dataset = lmdb_dataset
		
		# 0: ordinary order; 1: reverse order; 2: random order.
		# self.frame_order = frame_order
		# assert self.frame_order in [0, 1, 2]
		# 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
		self.slice_framepos = slice_framepos
		assert self.slice_framepos in [0, 1, 2]

		self.unfold_sentences = unfold_sentences
		self.sample_len = 0
		if self.unfold_sentences:
			# extract all sentences, i.e., 200000 text-video pairs
			train_video_ids = list(self.csv['video_id'].values)
			self.sentences_dict = {}
			for itm in self.data['sentences']:
				if itm['video_id'] in train_video_ids:
					self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
			self.sample_len = len(self.sentences_dict)
		else:
			num_sentences = 0
			self.sentences = defaultdict(list)
			s_video_id_set = set()
			for itm in self.data['sentences']:
				self.sentences[itm['video_id']].append(itm['caption'])
				num_sentences += 1
				s_video_id_set.add(itm['video_id'])

			# Use to find the clips in the same video
			self.parent_ids = {}
			self.children_video_ids = defaultdict(list)
			for itm in self.data['videos']:
				vid = itm["video_id"]
				url_posfix = itm["url"].split("?v=")[-1]
				self.parent_ids[vid] = url_posfix
				self.children_video_ids[url_posfix].append(vid)
			self.sample_len = len(self.csv)

		self.rawVideoExtractor = RawVideoExtractorpyAV(size=image_resolution,
														num_segments=self.max_frames,
														lmdb_dataset=self.lmdb_dataset)
		self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
							  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

	def __len__(self):
		return self.sample_len

	def _get_text(self, video_id, caption=None):
		"""get text"""
		k = 1
		choice_video_ids = [video_id]
		pairs_text = np.zeros((k, self.max_words), dtype=np.long)
		pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
		pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

		for i, video_id in enumerate(choice_video_ids):
			if caption is not None:
				words = self.tokenizer.tokenize(caption)
			else:
				words = self._get_single_text(video_id)

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

		return pairs_text, pairs_mask, pairs_segment, choice_video_ids

	def _get_single_text(self, video_id):
		rind = random.randint(0, len(self.sentences[video_id]) - 1)
		caption = self.sentences[video_id][rind]
		words = self.tokenizer.tokenize(caption)
		return words

	def _get_rawvideo(self, choice_video_ids):
		video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
		max_video_length = [0] * len(choice_video_ids)
		video_list = []
		# video data
		for i, video_id in enumerate(choice_video_ids):
			# torch.Tensor of shape [T, C, H, W]
			raw_video_data, slice_len = self.rawVideoExtractor.get_video_data(
									os.path.join(self.features_path, "{}.mp4".format(video_id))
									)
			# slice_len = raw_video_data.size(0)
			max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
			video_list.append(raw_video_data)
		# vide mask
		for i, v_length in enumerate(max_video_length):
			video_mask[i][:v_length] = [1] * v_length
		# torch.Tensor of shape [Pair, T, C, H, W]
		video = torch.stack(video_list, dim=0)

		return video, video_mask

	def __getitem__(self, idx):
		if self.unfold_sentences:
			video_id, caption = self.sentences_dict[idx]
		else:
			video_id, caption = self.csv['video_id'].values[idx], None
		# text shape [Pair (= 1), self.max_words]
		pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
		# video shape [Pair, T, C, H, W]
		video, video_mask = self._get_rawvideo(choice_video_ids)
		return pairs_text, pairs_mask, pairs_segment, video, video_mask
