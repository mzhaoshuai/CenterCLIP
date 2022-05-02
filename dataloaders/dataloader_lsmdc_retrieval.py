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


class LSMDC_DataLoader(Dataset):
	"""LSMDC dataset loader."""
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
		assert self.subset in ["train", "val", "test"]

		video_json_path_dict = {}
		video_json_path_dict["train"] = os.path.join(self.data_path, "LSMDC16_annos_training.csv")
		video_json_path_dict["val"] = os.path.join(self.data_path, "LSMDC16_annos_val.csv")
		video_json_path_dict["test"] = os.path.join(self.data_path, "LSMDC16_challenge_1000_publictect.csv")

		# <CLIP_ID>\t<START_ALIGNED>\t<END_ALIGNED>\t<START_EXTRACTED>\t<END_EXTRACTED>\t<SENTENCE>
		# <CLIP_ID> is not a unique identifier, i.e. the same <CLIP_ID> can be associated with multiple sentences.
		# However, LSMDC16_challenge_1000_publictect.csv has no repeat instances
		video_id_list = []
		caption_dict = {}
		with open(video_json_path_dict[self.subset], 'r') as fp:
			for line in fp:
				line = line.strip()
				line_split = line.split("\t")
				assert len(line_split) == 6
				clip_id, start_aligned, end_aligned, start_extracted, end_extracted, sentence = line_split
				caption_dict[len(caption_dict)] = (clip_id, sentence)
				if clip_id not in video_id_list: video_id_list.append(clip_id)
		# some videos are broken
		# broken_videos = ['1038_The_Great_Gatsby_01.22.15.059-01.22.21.087', 
		# 					'0014_Ist_das_Leben_nicht_schoen_01.28.37.413-01.28.53.788']
		broken_videos = []
		for m in broken_videos:
			if m in video_id_list: video_id_list.remove(m)

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

		# Get all captions
		self.iter2video_pairs_dict = {}
		for clip_id, sentence in caption_dict.values():
			if clip_id not in self.video_dict:
				continue
			self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (clip_id, sentence)

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
			timestamps = v_["timestamps"]
			pseudo_caption_dict[pseudo_video_id]["start"] = \
				np.array([int(math.floor(float(itm[0]))) for itm in timestamps], dtype=object)
			pseudo_caption_dict[pseudo_video_id]["end"] = \
				np.array([int(math.ceil(float(itm[1]))) for itm in timestamps], dtype=object)
			pseudo_caption_dict[pseudo_video_id]["text"] = np.array(v_["sentences"], dtype=object)
		return pseudo_caption_dict

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
			max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
			video_list.append(raw_video_data)

		# torch.Tensor of shape [Pair, T, C, H, W]
		video = torch.stack(video_list, dim=0)

		for i, v_length in enumerate(max_video_length):
			video_mask[i][:v_length] = [1] * v_length
		return video, video_mask

	def __getitem__(self, feature_idx):
		clip_id, sentence = self.iter2video_pairs_dict[feature_idx]
		pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(clip_id, sentence)
		video, video_mask = self._get_rawvideo(choice_video_ids)
		return pairs_text, pairs_mask, pairs_segment, video, video_mask


def generate_video_id_to_path_json(features_path='/home/shuai/dataset/lsmdc/video',
									output='/home/shuai/dataset/lsmdc/video_path.json'):
	video_dict = {}
	for root, dub_dir, video_files in os.walk(features_path):
		for video_file in video_files:
			video_id_ = ".".join(video_file.split(".")[:-1])
			# if video_id_ not in video_id_list:
			# 	continue
			file_path_ = os.path.join(root, video_file)
			relative_path = file_path_.replace(features_path, '')[1:]
			video_dict[video_id_] = relative_path
	
	with open(output, 'w') as f:
		json.dump(video_dict, f, indent=4, sort_keys=True)


if __name__ == "__main__":
	generate_video_id_to_path_json()
