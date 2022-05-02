# coding=utf-8
from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from .decode import RawVideoExtractorpyAV
# from dataloaders.rawvideo_util import RawVideoExtractor


class MSVD_DataLoader(Dataset):
	"""MSVD dataset loader."""
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
		"""
		In MSVD, one video have multiple descriptions.
		"""
		self.data_path = data_path
		self.features_path = features_path
		self.feature_framerate = feature_framerate
		self.max_words = max_words
		self.max_frames = max_frames
		self.tokenizer = tokenizer
		self.lmdb_dataset = lmdb_dataset

		self.subset = subset
		assert self.subset in ["train", "val", "test"]
		video_id_path_dict = {}
		video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
		video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
		video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")
		caption_file = os.path.join(self.data_path, "raw-captions.pkl")

		with open(video_id_path_dict[self.subset], 'r') as fp:
			video_ids = [itm.strip() for itm in fp.readlines()]

		with open(caption_file, 'rb') as f:
			captions = pickle.load(f)

		video_dict = {}
		for root, dub_dir, video_files in os.walk(self.features_path):
			for video_file in video_files:
				video_id_ = ".".join(video_file.split(".")[:-1])
				if video_id_ not in video_ids:
					continue
				file_path_ = os.path.join(root, video_file)
				video_dict[video_id_] = file_path_
		self.video_dict = video_dict

		self.sample_len = 0
		self.sentences_dict = {}
		self.cut_off_points = []
		for video_id in video_ids:
			assert video_id in captions
			for cap in captions[video_id]:
				cap_txt = " ".join(cap)
				self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
			self.cut_off_points.append(len(self.sentences_dict))

		## below variables are used to multi-sentences retrieval
		# self.cut_off_points: used to tag the label when calculate the metric
		# self.sentence_num: used to cut the sentence representation
		# self.video_num: used to cut the video representation
		self.multi_sentence_per_video = True    # !!! important tag for eval
		if self.subset == "val" or self.subset == "test":
			self.sentence_num = len(self.sentences_dict)
			self.video_num = len(video_ids)
			assert len(self.cut_off_points) == self.video_num
			print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
			print("For {}, video number: {}".format(self.subset, self.video_num))

		print("Video number: {}".format(len(self.video_dict)))
		print("Total Paire: {}".format(len(self.sentences_dict)))

		self.sample_len = len(self.sentences_dict)
		# self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
		self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
							  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

		self.rawVideoExtractor = RawVideoExtractorpyAV(size=image_resolution,
														is_train=(subset == 'train'),
														num_segments=self.max_frames,
														lmdb_dataset=self.lmdb_dataset)

	def __len__(self):
		return self.sample_len

	def _get_text(self, video_id, caption):
		k = 1
		choice_video_ids = [video_id]
		pairs_text = np.zeros((k, self.max_words), dtype=np.long)
		pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
		pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

		for i, video_id in enumerate(choice_video_ids):
			words = self.tokenizer.tokenize(caption)

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
			video_path = self.video_dict[video_id]
			raw_video_data, slice_len = self.rawVideoExtractor.get_video_data(video_path)
			# slice_len = raw_video_data.size(0)
			max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
			video_list.append(raw_video_data)
		
		# torch.Tensor of shape [Pair, T, C, H, W]
		video = torch.stack(video_list, dim=0)

		for i, v_length in enumerate(max_video_length):
			video_mask[i][:v_length] = [1] * v_length

		return video, video_mask

	def __getitem__(self, idx):
		video_id, caption = self.sentences_dict[idx]

		pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
		video, video_mask = self._get_rawvideo(choice_video_ids)
		return pairs_text, pairs_mask, pairs_segment, video, video_mask
