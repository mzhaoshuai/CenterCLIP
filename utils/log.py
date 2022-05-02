# coding=utf-8
# https://github.com/mlfoundations/open_clip/blob/bb665df657/src/training/logger.py
# https://gist.github.com/scarecrow1123/967a97f553697743ae4ec7af36690da6
import logging
import argparse
from logging import Filter
from logging.handlers import QueueHandler, QueueListener

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue


# def get_logger(filename=None):
#     """set logger"""
#     logger = logging.getLogger('logger')
#     logger.setLevel(logging.DEBUG)
#     logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
#                         datefmt='%m/%d/%Y %H:%M:%S',
#                         level=logging.INFO)
	
#     if filename is not None:
#         handler = logging.FileHandler(filename)
#         handler.setLevel(logging.DEBUG)
#         handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
#         logging.getLogger().addHandler(handler)
	
#     return logger


def setup_primary_logging(log_file, level, remote=False):
	"""
	Global logging is setup using this method. In a distributed setup, a multiprocessing queue is setup
	which can be used by the workers to write their log messages. This initializers respective handlers
	to pick messages from the queue and handle them to write to corresponding output buffers.
	Parameters
	----------
	log_file_path : ``str``, required
		File path to write output log
	Returns
	-------
	log_queue : ``torch.multiprocessing.Queue``
		A log queue to which the log handler listens to. This is used by workers
		in a distributed setup to initialize worker specific log handlers(refer ``setup_worker_logging`` method).
		Messages posted in this queue by the workers are picked up and bubbled up to respective log handlers.
	"""
	# Multiprocessing queue to which the workers should log their messages
	if not remote:
		log_queue = Queue(-1)
	else:
		mp = torch.multiprocessing.get_context('spawn')
		log_queue = mp.Queue(-1)

	# Handlers for stream/file logging
	file_handler = logging.FileHandler(filename=log_file)
	stream_handler = logging.StreamHandler()

	formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
									datefmt='%Y-%m-%d,%H:%M:%S')

	file_handler.setFormatter(formatter)
	stream_handler.setFormatter(formatter)

	file_handler.setLevel(level)
	stream_handler.setLevel(level)

	# This listener listens to the `log_queue` and pushes the messages to the list of
	# handlers specified.
	listener = QueueListener(log_queue, file_handler, stream_handler)

	listener.start()

	return log_queue


class WorkerLogFilter(Filter):
	def __init__(self, rank=-1):
		super().__init__()
		self._rank = rank

	def filter(self, record):
		# if self._rank != -1:
		# 	record.msg = f"Rank {self._rank} | {record.msg}"
		# return True

		# only log the msg of rank-0
		if self._rank == 0:
			record.msg = f"Rank {self._rank} | {record.msg}"
			return True
		else:
			return False


def setup_worker_logging(rank, log_queue, level=logging.INFO):
	"""
	Method to initialize worker's logging in a distributed setup. The worker processes
	always write their logs to the `log_queue`. Messages in this queue in turn gets picked
	by parent's `QueueListener` and pushes them to respective file/stream log handlers.
	Parameters
	----------
	rank : ``int``, required
		Rank of the worker
	log_queue: ``Queue``, required
		The common log queue to which the workers
	Returns
	-------
	features : ``np.ndarray``
		The corresponding log power spectrogram.
	"""
	queue_handler = QueueHandler(log_queue)

    # Add a filter that modifies the message to put the
    # rank in the log format
	worker_filter = WorkerLogFilter(rank)
	queue_handler.addFilter(worker_filter)

	queue_handler.setLevel(level)

	root_logger = logging.getLogger()
	root_logger.addHandler(queue_handler)

    # Default logger level is WARNING, hence the change. Otherwise, any worker logs
    # are not going to get bubbled up to the parent's logger handlers from where the
    # actual logs are written to the output
	root_logger.setLevel(level)


def fake_worker(rank: int, world_size: int, log_queue: Queue):
	setup_worker_logging(rank, log_queue, logging.DEBUG)
	logging.info("Test worker log")
	logging.error("Test worker error log")
	torch.cuda.set_device(rank)
	dist.init_process_group(
		backend='nccl',
		init_method='tcp://127.0.0.1:6100',
		world_size=world_size,
		rank=rank,
	)


if __name__ == "__main__":
	# Set multiprocessing type to spawn
	torch.multiprocessing.set_start_method("spawn")

	parser = argparse.ArgumentParser()
	parser.add_argument("-g", "--gpu-list", type=int, help="List of GPU IDs", nargs="+", required=True)

	args = parser.parse_args()

	world_size = len(args.gpu_list)

	# Initialize the primary logging handlers. Use the returned `log_queue`
	# to which the worker processes would use to push their messages
	log_queue = setup_primary_logging("/usr/lusers/gamaga/out.log", logging.DEBUG)

	if world_size == 1:
		worker(0, world_size, log_queue)
	else:
		mp.spawn(fake_worker, args=(world_size, log_queue), nprocs=world_size)

