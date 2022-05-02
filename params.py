#coding=utf-8

import os
import json
import logging
import argparse


def get_default_params(model_name):
	# Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
	if model_name in ["RN50", "RN101", "RN50x4"]:
		return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
	elif model_name in ["ViT-B/32", "ViT-B/16"]:
		return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
	else:
		return {}


def get_args(description='CenterCLIP on Retrieval Task'):
	"""config of program"""
	parser = argparse.ArgumentParser(description=description)

	parser.add_argument("--do_pretrain", action='store_true', default=False,
							help="Whether to run training.")
	parser.add_argument("--do_train", type=int, default=1,
							help="Whether to run training.")
	parser.add_argument("--do_eval", type=int, default=0,
							help="Whether to run eval on the dev set.")

	parser.add_argument("--inference_speed_test", type=int, default=0,
							help="Only test the inference speed.")

	parser.add_argument("--debug", default=False, action="store_true",
							help="If true, more information is logged.")
	# datasets
	parser.add_argument('--data_dir', type=str, default='/cache/dataset',
							help='where all data located')

	parser.add_argument('--lmdb_dataset', type=str, default=None,
							help="LMDB database for the dataset")

	parser.add_argument('--save_feature_path', type=str,
							default=None,
							help='Used to save the CLIP features')

	parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
	
	parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
	
	parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')

	parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

	# training settings
	parser.add_argument('--num_thread_reader', type=int, default=1, help='')

	parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')

	parser.add_argument('--batch_size', type=int, default=256, help='batch size')

	parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
	# learning strategies
	parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')

	parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')

	parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')

	parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")

	parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")

	parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")

	parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")

	parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
	parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
	parser.add_argument('--seed', type=int, default=42, help='random seed')
	parser.add_argument('--max_words', type=int, default=20, help='')
	parser.add_argument('--max_frames', type=int, default=100, help='')
	parser.add_argument('--feature_framerate', type=int, default=1, help='')
	parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
	parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
	parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
	parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

	parser.add_argument("--output_dir", default=None, type=str, required=True,
							help="The output directory where the model predictions and checkpoints will be written.")

	parser.add_argument("--resume", default=None, type=str,
							help="path to latest checkpoint (default: none)",)

	parser.add_argument('--load_from_pretrained', type=int, default=0,
							help="load optimizer and scaler state from pretrained model.")

	parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")

	parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")

	parser.add_argument("--do_lower_case", action='store_true',
							help="Set this flag if you are using an uncased model.")

	parser.add_argument("--optim", default='BertAdam', type=str, choices=['BertAdam', 'AdamW'],
							help="The optimizer")

	parser.add_argument("--warmup_proportion", default=0.1, type=float,
							help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")

	parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
							help="Number of updates steps to accumulate before performing a backward/update pass.")

	parser.add_argument("--clip_grad_norm", default=1.0, type=float,
							help="the maximum gradient norm (default None)")

	parser.add_argument("--cache_dir", default="", type=str,
							help="Where do you want to store the pre-trained models downloaded from s3")

	parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")

	parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

	parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
	parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

	parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
	parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
	parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

	parser.add_argument('--loose_type', action='store_true', 
							help="Default using tight type for retrieval.")
	parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

	parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
							help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
	parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
							help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

	parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
	parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
							help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
	parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
							help="linear projection of flattened patches.")
	parser.add_argument('--sim_header', type=str, default="meanP",
							choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
							help="choice a similarity header.")
	
	# setting about pretrained weights
	parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str,
							help="Choose a CLIP version")

	parser.add_argument("--pretrained_dir", default=os.path.expanduser("~/models/pretrained"), type=str,
							help="The pretrained directory of CLIP pretrained model")

	# arguments for distributed training
	parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")

	parser.add_argument('--world_size', default=1, type=int,
							help='number of nodes for distributed training')

	parser.add_argument("--local_rank", default=0, type=int, help="distribted training")

	parser.add_argument("--init_method", default="tcp://127.0.0.1:6101", type=str,
	   						help="url used to set up distributed training")

	# setting about GPUs
	parser.add_argument("--dp", default=False, action="store_true",
							help="Use DP instead of DDP.")

	parser.add_argument("--multigpu", default=None, type=lambda x: [int(a) for a in x.split(",")],
							help="In DP, which GPUs to use for multigpu training", )

	parser.add_argument("--gpu", type=int, default=None,
							help="Specify a single GPU to run the code on for debugging."
							 "Leave at None to use all available GPUs.", )

	parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

	parser.add_argument("--use-bn-sync", default=False, action="store_true",
							help="Whether to use batch norm sync.")

	# setting about remote server cluster
	parser.add_argument("--remote", type=int, default=0, help="use remote server cluster or not.")

	parser.add_argument("--data_loaded", type=int, default=0,
							help="already load data on remote server cluster.")

	# precision of training weights
	parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="fp32",
							help="Floating point precition.")

	# cluster algorithms
	parser.add_argument('--cluster_algo', type=str, default='kmediods++',
							choices=['kmediods++', 'pooling', 'sparse_sampling', 'spectral',
										'temporal_shift', 'token_shift'],
							help="The type of cluster algorithms.")

	parser.add_argument('--cluster_embedding', type=int, default=0,
							help="Whether using cluser embedding or not.")

	parser.add_argument('--cluser_embed_from_clip', type=int, default=1,
							help="Whether using CLIP pretrained positional embedding to initialize cluster embedding.")

	parser.add_argument('--cluster_frame_embedding', type=int, default=0,
							help="Whether using cluser frame embedding or not.")

	parser.add_argument('--adaptive_cls', type=int, default=0,
							help="Whether adaptive [CLASS] token fusion.")

	# parser.add_argument('--position_embed_first', type=int, default=0,
	# 						help="When clusttering, add position embedding first.")

	# parser.add_argument('--time_embed_frist', type=int, default=0,
	# 						help="When clustering, add frame embedding first.")

	parser.add_argument('--aggregation', type=str, default=None,
							choices=['mean', 'None'],
							help="When clustering, how to aggregate a cluster.")

	parser.add_argument('--cluster_iter_limit', type=int, default=100,
							help="Iteration limits of cluster algorithms.")

	parser.add_argument('--cluster_distance', type=str, default='euclidean',
							choices=['euclidean', 'cosine'],
							help="type of clustering distance.")

	parser.add_argument('--cluster_threshold', type=float, default=1e-5,
							help="stop threshold for clustering.")

	parser.add_argument('--minkowski_norm_p', type=float, default=2.0,
							help="p value for the p-norm distance to calculate between each vector pair.")

	# cluster algorithms -- for inter blocks clustering
	parser.add_argument('--cluster_inter', type=int, default=0,
							help="Whether use clustering algorithms inside transformer blocks.")	

	parser.add_argument('--cluster_num_blocks', type=int, default=0, nargs='+',
							help="The number of clusters in each transformer blocks.")

	parser.add_argument('--target_frames_blocks', type=int, default=[12] * 12, nargs='+',
							help="The target frames after clustering in each transformer blocks.")

	parser.add_argument('--spectral_sigma', type=float, default=2.0,
							help='Sigma of HeatKernel in Spectral clustering')

	parser.add_argument('--spectral_graph', type=str, default='HeatKernel',
							choices=['HeatKernel', 'KNN'],
							help="type of graph in spectral clustering.")

	parser.add_argument('--spectral_knn_k', type=int, default=1,
							help='K of KNN when constructing KNN graph in spectral learning, when value < 5,'
							'it will determined automatically')

	parser.add_argument('--spectral_spg', type=int, default=0,
							help="Spectral Temporal graph.")

	parser.add_argument('--svd_correct_sign', type=int, default=1,
							help="correct sign in SVD and PCA.")

	# cluster algorithms -- for deep clustering
	parser.add_argument('--deep_cluster', type=int, default=0,
							help="Whether use DeepCluster algorithm.")

	parser.add_argument('--cluster_inter_dim', type=int, default=256,
							help="Intermediate dimension of deep cluster model.")

	parser.add_argument('--freeze_clip', type=int, default=0,
							help="Whether freeze all clip backbone.")

	# divide the pretrained temperature
	parser.add_argument('--temperature_new', type=float, default=1.0,
							help='assign a new temperature to CLIP model')

	parser.add_argument('--time_embedding', type=int, default=0,
							help="Add time embedding in CLIP model.")

	# test of DSL loss in CAMOE
	parser.add_argument('--camoe_dsl', type=int, default=0,
							help="Add DSL loss for CAMOE.")

	parser.add_argument('--pre_norm', type=int, default=0,
							help="whether do l2 normalization before clustering.")

	args = parser.parse_args()

	assert args.task_type == "retrieval"
	assert not (args.deep_cluster and args.cluster_inter)

	if args.sim_header == "tightTransf":
		args.loose_type = False
	if args.datatype == 'activity':
		# pre-pooling to avoid OOM, only work for meanP with AcitivityNet when eval
		args.pre_visual_pooling = 1

	# Check paramenters
	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
			args.gradient_accumulation_steps))
	
	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir, exist_ok=True)

	args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

	args.tensorboard_path = os.path.join(args.output_dir, "tensorboard")
	# logging level
	args.log_level = logging.DEBUG if args.debug else logging.INFO

	# new added params
	# args.new_added_modules = ['time_embedding', 'frame_embedding']
	args.new_added_modules = ['time_embedding', 'frame_embedding', 'deepcluster']
	# args.new_added_modules = ['time_embedding', 'frame_embedding', 'deepcluster', 'tokencluster_inter']

	# If some params are not passed, we use the default values based on model name.
	default_params = get_default_params(args.pretrained_clip_name)
	for name, val in default_params.items():
		if getattr(args, name) is None:
			setattr(args, name, val)

	print('\n', vars(args), '\n')
	# save_hp_to_json(args.output_dir, args)

	return args


def save_hp_to_json(directory, args):
	"""Save hyperparameters to a json file
	"""
	filename = os.path.join(directory, 'hparams_train.json')
	hparams = vars(args)
	with open(filename, 'w') as f:
		json.dump(hparams, f, indent=4, sort_keys=True)


if __name__ == "__main__":
	args = get_args()
