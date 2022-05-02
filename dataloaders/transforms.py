# coding=utf-8
# some image/video transforms for PIL.Image or torch.Tensor
import torch
import random
import torchvision
import numpy as np


class Div255(object):
	"""converts a torch.Tensor in the range [0.0, 1.0]"""
	def __init__(self, div=True):
		super(Div255).__init__()
		self.div = div

	def __call__(self, tensor):
		return tensor.float().div_(255.0) if self.div else tensor.float()


class TensorNormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		"""
		Args:
			x / sample['frames'] (torch.Tensor): video tensor with shape (T, C, H, W).
		"""
		rep_mean = self.mean * (tensor.size()[1] // len(self.mean))
		rep_std = self.std * (tensor.size()[1] // len(self.std))

		tensor = torchvision.transforms.functional.normalize(tensor, mean=rep_mean, std=rep_std, inplace=True)

		return tensor


class TensorMultiScaleCrop(object):

	def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True,
						more_fix_crop=True,
						interpolation=torchvision.transforms.InterpolationMode.BILINEAR):
		"""
		Args:
		interpolation:
			Desired interpolation enum defined by torchvision.transforms.InterpolationMode.
			Default is InterpolationMode.BILINEAR.
			If input is Tensor, only InterpolationMode.NEAREST, InterpolationMode.BILINEAR and InterpolationMode.BICUBIC 
			are supported.
			For backward compatibility integer values (e.g. PIL.Image.NEAREST) are still acceptable.
		"""
		self.scales = scales if scales is not None else [1, .875, .75, .66]
		self.max_distort = max_distort
		self.fix_crop = fix_crop
		self.more_fix_crop = more_fix_crop
		self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
		self.interpolation = interpolation

	def __call__(self, sample):
		"""
		Args:
			x / sample['frames'] (torch.Tensor): video tensor with shape (T, C, H, W).
		"""
		if isinstance(sample, dict):
			x, label = sample['frames'], sample['label']
		else:
			x = sample
		assert len(x.shape) == 4
		# assert x.dtype == torch.float32

		# the width, height of the tensor
		im_size = (x.shape[3], x.shape[2])
		crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
		# (top, left, height, width)
		crop_x = torchvision.transforms.functional.resized_crop(x, offset_h, offset_w,
								crop_h, crop_w, size=self.input_size, interpolation=self.interpolation)

		if isinstance(sample, dict):
			return {'frames': crop_x,
					'label': label}
		else:
			return crop_x

	def _sample_crop_size(self, im_size):
		image_w, image_h = im_size[0], im_size[1]

		# find a crop size
		base_size = min(image_w, image_h)
		crop_sizes = [int(base_size * x) for x in self.scales]
		crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
		crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

		pairs = []
		for i, h in enumerate(crop_h):
			for j, w in enumerate(crop_w):
				if abs(i - j) <= self.max_distort:
					pairs.append((w, h))

		crop_pair = random.choice(pairs)
		if not self.fix_crop:
			w_offset = random.randint(0, image_w - crop_pair[0])
			h_offset = random.randint(0, image_h - crop_pair[1])
		else:
			w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

		return crop_pair[0], crop_pair[1], w_offset, h_offset

	def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
		offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
		return random.choice(offsets)

	@staticmethod
	def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
		w_step = (image_w - crop_w) // 4
		h_step = (image_h - crop_h) // 4

		ret = list()
		ret.append((0, 0))                      # upper left
		ret.append((4 * w_step, 0))             # upper right
		ret.append((0, 4 * h_step))             # lower left
		ret.append((4 * w_step, 4 * h_step))    # lower right
		ret.append((2 * w_step, 2 * h_step))    # center

		if more_fix_crop:
			ret.append((0, 2 * h_step))             # center left
			ret.append((4 * w_step, 2 * h_step))    # center right
			ret.append((2 * w_step, 4 * h_step))    # lower center
			ret.append((2 * w_step, 0 * h_step))    # upper center

			ret.append((1 * w_step, 1 * h_step))    # upper left quarter
			ret.append((3 * w_step, 1 * h_step))    # upper right quarter
			ret.append((1 * w_step, 3 * h_step))    # lower left quarter
			ret.append((3 * w_step, 3 * h_step))    # lower righ quarter

		return ret


class GroupToTensorBCHW(object):
	"""
	Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
	to a torch.FloatTensor of shape (H x W x C) in the range [0.0, 1.0]
	"""
	def __init__(self, div=True):
		self.div = div

	def __call__(self, sample):
		if isinstance(sample, dict):
			img_group, label = sample['frames'], sample['label']
			return {'frames': torch.stack([self._toTensor(x) for x in img_group], dim=0),
					'label': label}
		else:
			return torch.stack([self._toTensor(x) for x in sample], dim=0)

	def _toTensor(self, pic):
		if isinstance(pic, np.ndarray):
			# handle numpy array
			img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
			#img = torch.from_numpy(pic).contiguous()
		else:
			# handle PIL Image
			img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
			img = img.view(pic.size[1], pic.size[0], len(pic.mode))
			# put it from HWC to CHW format
			# yikes, this transpose takes 80% of the loading time/CPU
			img = img.transpose(0, 1).transpose(0, 2).contiguous()
		return img.float().div_(255) if self.div else img.float()


class TensorRandomHorizontalFlip(object):
	"""Randomly horizontally flips the given PIL.Image with a probability of 0.5
	"""
	def __init__(self, is_flow=False):
		self.is_flow = is_flow

	def __call__(self, sample, is_flow=False):
		"""
		Args:
			x / sample['frames'] (torch.Tensor): video tensor with shape (T, C, H, W).
		"""
		v = random.random()
		if v < 0.5:
			if isinstance(sample, dict):			
				x, label = sample['frames'], sample['label']
			else:
				x = sample
			flip_x = torchvision.transforms.functional.hflip(x)
			if self.is_flow:
				# invert the color value of flow_x
				flip_x[:, ::2, ...] = torchvision.transforms.functional.invert(flip_x[:, ::2, ...])
			
			if isinstance(sample, dict):
				return {'frames': flip_x,
						'label': label}
			else:
				return flip_x
		else:
			return sample

