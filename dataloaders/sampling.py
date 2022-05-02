# coding=utf-8
"""tools for frame sampling
"""
import numpy as np
from numpy.random import randint


def multi_segments_sampling(clip_length, num_frames, random_shift=True, modality="RGB", data_length=1):
	"""
	A TSN style multiple segments sampling methods.
	The video is divided insto serveral segments and then we sample one frame randomly/uniformly
	from each of the segment.
	A video: 								1 2 3 4 5 6 7 8 9 10 11 12
	3 Segment:								1 2 3 4|5 6 7 8|9 10 11 12
	randomly select from each segment:		1 	   |      8|  10	
	-----------------------------------------
	Args:
		clip_length: the target sampling frames
		num_frames: the total frames of the video
		data_length: how many frames will be sampling at a time,
						the sampling will be done {clip_length} times 
	Return:
		return offsets of the frames (starts from 0)
	"""
	if random_shift:
		# add random perturbation to the frame selection
		average_duration = (num_frames - data_length + 1) // clip_length
		if average_duration > 0:
			offsets = np.multiply(list(range(clip_length)), average_duration) + \
						randint(average_duration, size=clip_length)

		elif num_frames > clip_length:
			# offsets = np.sort(randint(num_frames - data_length + 1, size=clip_length))
			# this line suits for data_length = 1
			offsets = np.sort(np.random.choice(num_frames, clip_length, replace=False))
		else:
			offsets = np.clip(np.arange(0, clip_length * data_length, data_length),
									0, num_frames - data_length)

	else:
		# general uniform sampling
		if num_frames > clip_length + data_length - 1:
			tick = (num_frames - data_length + 1) / float(clip_length)
			# center of the segment, tick / 2.0
			offsets = [int(tick / 2.0 + tick * x) for x in range(clip_length)]
			offsets = np.array(offsets)
		else:
			offsets = np.clip(np.arange(0, clip_length * data_length, data_length),
										0, num_frames - data_length)

	return offsets


def uniform_sampling(clip_length,
						num_frames,
						data_length=1,
						twice_sample=False):
	"""
	evenly sampling some frames from a video.
	-----------------------------------------	
	Args:
		clip_length: the target sampling frames
		num_frames: the total frames of the video

	Return:
		return offsets of the frames (starts from 0)
	"""
	if num_frames > clip_length + data_length - 1:
		tick = (num_frames - data_length + 1) / float(clip_length)
		# general uniform sampling
		if twice_sample:	
			offsets = ([int(tick / 2.0 + tick * x) for x in range(clip_length)] +
							[int(tick * x) for x in range(clip_length)])
		else:
			offsets = ([int(tick / 2.0 + tick * x) for x in range(clip_length)])
	else:
			offsets = np.clip(np.arange(0, clip_length * data_length, data_length),
										0, num_frames - data_length)
	offsets = np.array(offsets)	

	return offsets


if __name__ == "__main__":
	inds = multi_segments_sampling(60, 24, random_shift=False, modality="RGB", data_length=1)
	print(inds)
	inds = uniform_sampling(60, 24, data_length=1, twice_sample=False)
	print(inds)
