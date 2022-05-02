# coding

import os
import math
import json
import torch
import shutil
import random
import logging
import datetime
import numpy as np


def save_checkpoint(state, is_best, model_dir, filename='checkpoint.pth.tar'):
	filename = os.path.join(model_dir, filename)
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def save_model(epoch, args, model, type_name=""):
	# Only save the model it-self
	model_to_save = model.module if hasattr(model, 'module') else model
	output_model_file = os.path.join(args.output_dir,
										"pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
	torch.save(model_to_save.state_dict(), output_model_file)
	print("Model saved to %s", output_model_file)
	
	return output_model_file


# def load_model(epoch, args, n_gpu, device, model_file=None):
# 	if model_file is None or len(model_file) == 0:
# 		model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
# 	if os.path.exists(model_file):
# 		model_state_dict = torch.load(model_file, map_location='cpu')
# 		if args.local_rank == 0:
# 			logger.info("Model loaded from %s",'%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name) model_file)
# 		# Prepare model
# 		cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
# 		model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

# 		model.to(device)
# 	else:
# 		model = None
# 	return model


def set_random_seed(seed=None):
	"""set random seeds for pytorch, random, and numpy.random
	"""
	if seed is not None:
		os.environ['PYTHONHASHSEED'] = str(seed)
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)


def get_the_number_of_params(model, is_trainable=False):
	"""get the number of the model"""
	if is_trainable:
		return sum(p.numel() for p in model.parameters() if p.requires_grad)
	return sum(p.numel() for p in model.parameters())


# Used by https://github.com/openai/CLIP/issues/83 but not below.
# Keeping it incase needed.
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

