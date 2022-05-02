# coding=utf-8
"""
read videos from a folder and create a lmdb dataset
"""
import os
import lmdb
import argparse
from tqdm import tqdm


def video_folder2lmdb(folder_path, db_path, db_name, prefix='', video_list=None):
	"""
	Convert discontinuous files into a continuous LMDB database
	"""
	print('parse video under folder {}'.format(folder_path))
	video_folders = []
	if video_list is None:
		# for video in os.listdir(folder_path):
		# 	video_folders.append(video)
		# walk through all sub-folders
		for root, dub_dir, video_files in os.walk(folder_path):
			for video_file in video_files:
				video_folders.append(os.path.join(root, video_file))	
	else:
		video_folders = video_list

	lmdb_path = os.path.join(db_path, "{}.lmdb".format(db_name))
	isdir = os.path.isdir(lmdb_path)

	print("Generate LMDB to {}".format(lmdb_path))
	db = lmdb.open(lmdb_path, subdir=isdir,
				   map_size=1 << 43,
				   readonly=False,
				   meminit=False, map_async=True)

	# traverse over all folders and write the image data into the database
	txn = db.begin(write=True)
	for video in tqdm(video_folders):
		# read raw video data and store to db
		with open(video, mode='rb') as file:
			video_data = file.read()
		basename = os.path.basename(video)
		key = basename.encode()
		flag = txn.put(key, video_data)
		if not flag:
			raise IOError("LMDB write error!")

	txn.commit()
	db.sync()
	db.close()


def lmdb_video_decode_test(db_path, key, output_path):
	"""test for the video lmdb dataset"""
	import av
	import io

	env = lmdb.open(db_path,
						subdir=os.path.isdir(db_path),
						readonly=True, lock=False,
						readahead=False, meminit=False)
	with env.begin(write=False) as txn:
		data = txn.get(key.encode())
		container = av.open(io.BytesIO(data))

	for frame in container.decode(video=0):
		filename = os.path.join(output_path, 'frame-%04d.jpg' % frame.index)
		frame.to_image().save(filename) 	


if __name__ == "__main__":  
	parser = argparse.ArgumentParser(description='create a video lmdb dataset')
	parser.add_argument('--video_folder', type=str,
							default=os.path.expanduser('~/dataset1/activitynet/act_resized'),
							help='the video folders')
	parser.add_argument('--lmdb_path', type=str,
							default=os.path.expanduser('~/dataset1/activitynet/lmdb'),
							help='the path of the output LMDB dataset')
	parser.add_argument('--output_path', type=str,
							default=os.path.expanduser('~/output'),
							help='the output path')
	parser.add_argument('--lmdb_name', type=str, default='activity',
							help='the name of LMDB dataset')
	args = parser.parse_args()

	video_folder2lmdb(args.video_folder, args.lmdb_path, args.lmdb_name,
						prefix='', video_list=None)

	# lmdb_video_decode_test(os.path.join(args.lmdb_path, 'msrvtt.lmdb'), 
	# 						'video2500.mp4',
	# 						args.output_path)
