# coding=utf-8
import os
import av
import sys
import numpy as np
from PIL import Image
from torchvision.transforms.functional import center_crop


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


def img_seg(folder, n_px=224):
	"""divide image to square patches, used for the Figure 1 in the paper"""
	files = os.listdir(folder)
	# patch size
	w = 32
	stride = 32

	for file in files:
		fileid, ext = os.path.splitext(file)
		sub_folder = os.path.join(folder, fileid)
		if not os.path.exists(sub_folder):
			os.mkdir(sub_folder)

		img = Image.open(os.path.join(folder, file))
		# img = center_crop(img, n_px)
		# img.save(os.path.join(folder, fileid + '-' + 'crop' + ext))

		hight, width = img.size
		cnt = 1
		i = 0
		while (i + w <= hight):
			j = 0
			while (j + w <= width):
				new_img = img.crop((i, j, i + w, j + w))
				new_filename = os.path.join(sub_folder, fileid + '-' + str(cnt) + ext)
				new_img.save(new_filename)

				cnt += 1
				j += stride
			i = i + stride


def video_sample_patches(video_path, output_path, max_frames=12, npx=224):
	"""sample patches from video, used for the Figure 1 in the paper"""
	container = av.open(video_path)
	video_stream = container.streams.video[0]
	num_frames, fps = video_stream.frames, float(video_stream.average_rate)

	# pyav decode -- extract video frames from video stream
	sampled_frames, all_frames = [], []
	for frame in container.decode(video=0):
		all_frames.append(frame)

	inds = uniform_sampling(max_frames, num_frames, twice_sample=False)
	
	# av.Frames --> np.ndarray of shape [H, W, C]
	try:
		sampled_frames = [all_frames[i].to_image() for i in inds]
	except IndexError:
		print('The num_frames / fps are {} / {}, the inds are {}, '
				'the length of all frames are {}'.format(num_frames, fps, inds, len(all_frames)))
		print('The video file is {}'.format(video_path))
		sys.exit(1)

	crop_sampled_frames = [center_crop(x, npx) for x in sampled_frames]
	print('The number of frames are {}'.format(len(crop_sampled_frames)))

	# save original frames
	for i, im in enumerate(crop_sampled_frames):
		filename = os.path.join(output_path, 'crop-' + str(i + 1) + '.png')
		im.save(filename)

	# cut images
	img_seg(output_path, n_px=npx)


if __name__ == '__main__':
	# video_path = '/home/zhaoshuai/output/1054_Harry_Potter_and_the_prisoner_of_azkaban_00.00.48.396-00.00.53.175.avi'
	# output_path = '/home/zhaoshuai/output/harry908'
	video_path = '/home/zhaoshuai/output/1054_Harry_Potter_and_the_prisoner_of_azkaban_00.32.25.988-00.32.31.142.avi'
	output_path = '/home/zhaoshuai/output/harry654'

	video_sample_patches(video_path, output_path, max_frames=12, npx=224)

