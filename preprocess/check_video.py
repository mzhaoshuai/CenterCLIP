# coding=utf-8
import os
import av
import sys

def traverse_video_folder(folder_path):
	"""
	traverse a video folder and try to check them,
	this function does not work well now
	"""
	print('parse video under folder {}'.format(folder_path))
	for root, dub_dir, video_files in os.walk(folder_path):
		for video_file in video_files:
			video_filename = os.path.join(root, video_file)
			try:
				container = av.open(video_filename)
				print(video_filename)
				video_stream = container.streams.video[0]
				num_frames, fps = video_stream.frames, float(video_stream.average_rate)
				# if num_frames < 1:
				# 	os.remove(video_filename)
				# 	raise IOError("video {} is broken".format(video_filename))

			except av.AVError:
				print("video path: {} error".format(video_filename))
				# os.remove(video_filename)


def traverse_video_folder_cv2(folder_path):
	"""
	traverse a video folder and try to check them
	"""
	import cv2
	print('parse video under folder {}'.format(folder_path))
	for root, dub_dir, video_files in os.walk(folder_path):
		for video_file in video_files:
			video_filename = os.path.join(root, video_file)
			# produce error
			try:
				vid = cv2.VideoCapture(video_filename)
				if not vid.isOpened():
					raise IOError("video {} is broken".format(video_filename))
			except cv2.error as e:
				print("cv2.error:", e)
			except Exception as e:
				print("Exception:", e)
			else:
				pass


if __name__ == "__main__":
	traverse_video_folder_cv2('/home/shuai/dataset/lsmdc/video')
