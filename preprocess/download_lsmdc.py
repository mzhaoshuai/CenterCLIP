# coding=utf-8
"""
download LSMDC dataset
"""

import os
import subprocess
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def mp4_downloader(url, output_path='', only_down=False, user='xyz', passwd='uvw'):
	"""
	download video from url and resize (optional, disable with only_down=True),
	username and password from lsmdc official are needed.
	"""

	split_tmp = url.split('/')
	dirs, filename = split_tmp[-2], split_tmp[-1]
	if not os.path.exists(os.path.join(output_path, dirs)):
		os.mkdir(os.path.join(output_path, dirs))

	output_filename = os.path.join(output_path, dirs, filename)

	if only_down:
		cmd = 'wget -c {} -O {} --user={} --password={}'.format(
				url, output_filename, user, passwd)
		subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)      

	else:
		if os.path.exists(output_filename):
			return 1
		else:
			# download
			f_, e_ = os.path.splitext(filename)     
			temp_filename = os.path.join(output_path, dirs, filename.replace(e_, '_temp' + e_))

			cmd = 'wget -c {} -O {} --user={} --password={} --no-check-certificate'.format(
							url, temp_filename, user, passwd)
			# print(cmd)
			subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			# resize
			try:
				# https://trac.ffmpeg.org/ticket/309
				command = ['ffmpeg',
						'-y',                    # (optional) overwrite output file if it exists
						'-i', temp_filename,
						'-filter:v',
						'scale=\'if(gt(a,1),trunc(oh*a/2)*2,224)\':\'if(gt(a,1),224,trunc(ow*a/2)*2)\'',  # scale to 224
						'-map', '0:v',
						'-r', '3',              # frames per second
						output_filename,
						]
				ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				out, err = ffmpeg.communicate()
				retcode = ffmpeg.poll()
				# print something above for debug
				os.remove(temp_filename)
			
			except Exception as e:
				print(url)
				raise e


def mp_lsmdc_download(txt_file='.', output_path='.', n_thread=8, only_down=False, user='xyz', passwd='uvw'):
	"""downlaod lsmdc
	"""
	with open(txt_file) as f:
		content = f.readlines()
	urls = [x.strip() for x in content if 'http' in x]
	urls = sorted(set(urls))
	print('The total number of the url is {}'.format(len(urls)))

	# sys.exit(0)
	p = Pool(n_thread)
	worker = partial(mp4_downloader, output_path=output_path, only_down=only_down, user=user, passwd=passwd)
	for _ in tqdm(p.imap_unordered(worker, urls), total=len(urls)):
		pass
	p.close()
	p.join()

	print('\n')


if __name__ == "__main__":
	home = os.path.expanduser('~')
	print(home)
	# step 1: download the three files from https://sites.google.com/site/describingmovies/download and
	# place them in the same directory like the code does
	txt_file_list = [os.path.join(home, "dataset1/lsmdc", x)
						for x in ['MPIIMD_downloadLinks.txt', 'BlindTest_downloadLinks.txt', 'MVADaligned_downloadLinks.txt']]
	print(txt_file_list)

	# step 2: set the output directory for video data
	output_path = os.path.join(home, "dataset1/lsmdc/video")
	print(output_path)

	# step 3: download lsmdc video with lsmdc username and passwd
	n_thread = 100
	for txt_file in txt_file_list:
		mp_lsmdc_download(txt_file, output_path=output_path, n_thread=n_thread,
							only_down=False, user='xyz', passwd='uvw')
