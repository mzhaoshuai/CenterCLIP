#!/bin/bash
export PYTHONUNBUFFERED="True"

# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "Using local machine for training"

# dataset
dataset=activity
fps=3

data_dir=${HOME}/dataset1/data_activity
DATA_PATH=${data_dir}/activity
data_path=${DATA_PATH}
features_path=${DATA_PATH}/act_resized
pretrained_dir=${data_dir}/pretrained
# lmdb_dataset=${DATA_PATH}/lmdb/activity.lmdb
lmdb_dataset=None
train_csv=${DATA_PATH}/MSRVTT_train.7k.csv
data_loaded=0

# train or eval
do_train=1
do_eval=0


# learning strategies
pretrained_clip_name=ViT-B/32
lr=1e-2
coef_lr=1e-3
wd=0.2
epochs=8
optim=AdamW
num_workers=6
# FOR DiDeMo and ActivityNet, use more words and video frames
max_words=77
max_frames=60
temperature_new=1.0
resume=None
time_embedding=0
batch_size=16           # single GPU batch size
n_display=50            # log per n_display
precision=amp
freeze_clip=0


# token cluster inter
cluster_inter=0
cluster_algo='kmediods++'
cluster_embedding=0
cluster_distance='euclidean'
minkowski_norm_p=2.0
cluster_iter_limit=100
cluster_threshold=1e-6
cluster_frame_embedding=0
spectral_sigma=2.0
spectral_graph='HeatKernel'
spectral_knn_k=1


# For ViT-B/32
# plan 1. B_6 -- 20, 49
cluster_num_blocks='49 49 49 49 49 49 49 49 49 49 49 49'
target_frames_blocks='60 60 60 60 60 60 20 20 20 20 20 20'
# plan 2. B_6 -- 15, 49
target_frames_blocks='60 60 60 60 60 60 15 15 15 15 15 15'

# For ViT-B/16
# plan 1. B_6 -- 20, 160
# cluster_num_blocks='196 196 196 196 196 196 160 160 160 160 160 160'

# Perform token_shift at every blocks
# cluster_algo='token_shift'
# target_frames_blocks='11 11 11 11 11 11 11 11 11 11 11 11'
# cluster_num_blocks='55 54 53 52 51 50 48 47 46 45 44 43'

# distributed training
init_method='tcp://127.0.0.1:6061'

# run file
runfile=../main.py 


for num in 06
do 
	case ${num} in
		01 )
			# 32 V100 32GB GPUs
			pretrained_clip_name=ViT-B/16
			lr=1e-2
			cluster_inter=0
			batch_size=4
			max_words=77
			max_frames=60
			;;
		02 )
			pretrained_clip_name=ViT-B/16
			lr=1e-2
			batch_size=4
			cluster_inter=1
			cluster_algo='kmediods++'
			cluster_num_blocks='196 196 196 196 196 196 160 160 160 160 160 160'
			target_frames_blocks='60 60 60 60 60 60 20 20 20 20 20 20'
			;;
		03 )
			pretrained_clip_name=ViT-B/16
			lr=1e-2
			batch_size=4
			cluster_inter=1
			cluster_algo='kmediods++'
			cluster_num_blocks='196 196 196 196 196 196 160 160 160 160 160 160'
			target_frames_blocks='60 60 60 60 60 60 15 15 15 15 15 15'
			;;
		05 )
			pretrained_clip_name=ViT-B/16
			lr=1e-2
			batch_size=4
			cluster_inter=1
			cluster_algo='kmediods++'
			cluster_num_blocks='196 196 196 196 196 196 160 160 160 160 160 160'
			target_frames_blocks='60 60 60 60 60 60 12 12 12 12 12 12'
			;;
		04 )
			# 8 V100 32GB GPUs
			pretrained_clip_name=ViT-B/32
			lr=1e-2
			batch_size=16
			cluster_inter=1
			cluster_algo='token_shift'
			target_frames_blocks='11 11 11 11 11 11 11 11 11 11 11 11'
			cluster_num_blocks='55 54 53 52 51 50 48 47 46 45 44 43'
			;;
		06 )
			pretrained_clip_name=ViT-B/32
			lr=1e-2
			batch_size=16
			cluster_inter=1
			cluster_algo='kmediods++'
			max_words=77
			max_frames=75
			cluster_num_blocks='49 49 49 49 49 49 49 49 49 49 49 49'
			target_frames_blocks='75 75 75 75 75 75 15 15 15 15 15 15'
			;;
		07 )
			pretrained_clip_name=ViT-B/32
			lr=1e-2
			batch_size=16
			cluster_inter=1
			cluster_algo='kmediods++'
			max_words=77
			max_frames=45
			cluster_num_blocks='49 49 49 49 49 49 49 49 49 49 49 49'
			target_frames_blocks='45 45 45 45 45 45 15 15 15 15 15 15'
			;;
		08 )
			pretrained_clip_name=ViT-B/32
			lr=1e-2
			batch_size=16
			cluster_inter=1
			cluster_algo='kmediods++'
			max_words=77
			max_frames=30
			cluster_num_blocks='49 49 49 49 49 49 49 49 49 49 49 49'
			target_frames_blocks='30 30 30 30 30 30 15 15 15 15 15 15'
			;;
		* )
			;;
	esac

model_dir=${HOME}/models/eclip/eclip_${dataset}_${num}
echo "The model dir is ${model_dir}"

python ${runfile} \
		--do_train ${do_train} \
		--do_eval ${do_eval} \
		--num_thread_reader ${num_workers} \
		--epochs ${epochs} \
		--batch_size ${batch_size} \
		--n_display ${n_display} \
		--data_dir ${data_dir} \
		--lmdb_dataset ${lmdb_dataset} \
		--train_csv ${train_csv} \
		--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
		--data_path ${data_path} \
		--features_path ${features_path} \
		--output_dir ${model_dir} \
		--optim ${optim} \
		--lr ${lr} \
		--coef_lr ${coef_lr} \
		--wd ${wd} \
		--max_words ${max_words} \
		--max_frames ${max_frames} \
		--batch_size_val 16 \
		--datatype ${dataset} \
		--expand_msrvtt_sentences  \
		--feature_framerate ${fps} \
		--freeze_layer_num 0  \
		--slice_framepos 2 \
		--loose_type \
		--linear_patch 2d \
		--sim_header meanP \
		--pretrained_clip_name ${pretrained_clip_name} \
		--precision ${precision} \
		--data_loaded ${data_loaded} \
		--init_method ${init_method} \
		--pretrained_dir ${pretrained_dir} \
		--cluster_algo ${cluster_algo} \
		--cluster_threshold ${cluster_threshold} \
		--cluster_distance ${cluster_distance} \
		--minkowski_norm_p ${minkowski_norm_p} \
		--cluster_iter_limit ${cluster_iter_limit} \
		--temperature_new ${temperature_new} \
		--cluster_inter ${cluster_inter} \
		--cluster_embedding ${cluster_embedding} \
		--cluster_frame_embedding ${cluster_frame_embedding} \
		--cluster_num_blocks ${cluster_num_blocks} \
		--target_frames_blocks ${target_frames_blocks} \
		--spectral_sigma ${spectral_sigma} \
		--spectral_graph ${spectral_graph} \
		--spectral_knn_k ${spectral_knn_k} \
		--freeze_clip ${freeze_clip} \
		--resume ${resume}

done
echo "Finish Training"
