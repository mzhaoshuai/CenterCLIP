# coding=utf-8
"""generate video path json for LSMDC and ActivityNet"""
import os
import json

def generate_video_id_to_path_json(features_path='/home/zhaoshuai/dataset1/lsmdc/video',
									output='/home/zhaoshuai/dataset1/lsmdc/video_path.json'):
	video_dict = {}
	for root, dub_dir, video_files in os.walk(features_path):
		for video_file in video_files:
			video_id_ = ".".join(video_file.split(".")[:-1])
			file_path_ = os.path.join(root, video_file)
			relative_path = file_path_.replace(features_path, '')[1:]
			video_dict[video_id_] = relative_path

	with open(output, 'w') as f:
		json.dump(video_dict, f, indent=4, sort_keys=True)


if __name__ == "__main__":
	generate_video_id_to_path_json()

