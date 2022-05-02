# coding=utf-8
import os
import av
import sys
import numpy as np
from PIL import Image, ImageDraw
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


def visualize_image(video_path, id_path, output_path,
                        max_frames=12, frame_duration=2, npx=224,
                        patch_hw=7):
    """
    visualize the clustered image, id start from 0.
    Used for the token visualization Figure in the paper"""
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

    # read reserved id
    with open(id_path, 'rb') as f:
        cluster_ids = np.load(f)
    print('The read ids is {}'.format(cluster_ids))

    # masked image
    step = 32
    total_per_frame = patch_hw ** 2
    all_ids = []
    for i in range(len(crop_sampled_frames)):
        draw = ImageDraw.Draw(crop_sampled_frames[i], 'RGBA')
        t_tmp = i % frame_duration
        segment = i // frame_duration

        for y in range(patch_hw):
            for x in range(patch_hw):
                id_tmp = t_tmp * total_per_frame + y * patch_hw + x
                if id_tmp not in cluster_ids[segment, :]:
                    draw.rectangle([x * step, y * step, (x + 1) * step, (y + 1) * step], fill=(192, 192, 192, 170))
                else:
                    all_ids.append(id_tmp)

        filename = os.path.join(output_path, 'crop-masked-' + str(i + 1) + '.png')
        crop_sampled_frames[i].save(filename)

    print("all ids are {}".format(all_ids))


if __name__ == '__main__':
    video_path = '/home/zhaoshuai/output/1054_Harry_Potter_and_the_prisoner_of_azkaban_00.36.05.525-00.36.13.349.avi'
    output_path = '/home/zhaoshuai/output/1054_harry_id_f2'
    id_path = '/home/zhaoshuai/output/lsmdc_id/ids_30.npy'

    frame_duration = 2
    visualize_image(video_path, id_path, output_path,
                            max_frames=12, frame_duration=frame_duration, npx=224,
                            patch_hw=7)
