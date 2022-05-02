#!/bin/bash
export PYTHONUNBUFFERED="True"

# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "Using local machine for training"

# dataset
dataset=lsmdc
fps=3

DATA_PATH=${HOME}/dataset1/lsmdc
features_path=${DATA_PATH}/video
data_path=${DATA_PATH}
train_csv=${DATA_PATH}/MSRVTT_train.7k.csv 
# where the CLIP pretrained weights located
pretrained_dir=${HOME}/models/pretrained
# lmdb_dataset=${DATA_PATH}/lmdb/lsmdc.lmdb
lmdb_dataset=None

# train or eval
do_train=1
do_eval=0

# learning strategies
pretrained_clip_name=ViT-B/32
lr=5e-3
coef_lr=1e-3
wd=0.2
epochs=5
optim=AdamW
# FOR DiDeMo and ActivityNet, use more words and video frames
# FOR MSR-VTT, MSVD, LSMDC, using 32 words and 12 frames is enough 
max_words=32
max_frames=12
temperature_new=1.0
resume=None
load_from_pretrained=0


batch_size=64           # single GPU batch size
batch_size_val=32
num_workers=8
n_display=40            # log per n_display
precision=amp
# precision='fp32'
# precision='fp16'
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
pre_norm=0

# token cluster inter, configs for ViT-B/32
cluster_num_blocks='49 49 49 49 49 49 49 49 49 49 49 49'
target_frames_blocks='12 12 12 12 12 12 6 6 6 6 6 6'

# configs for ViT-B/16
# cluster_num_blocks='196 196 196 196 196 196 160 160 160 160 160 160'
# target_frames_blocks='12 12 12 12 12 12 4 4 4 4 4 4'

# distributed training
init_method='tcp://127.0.0.1:6061'
save_feature_path=None


for num in 04 09
do
	case ${num} in
		01 )
			lr=5e-3
			optim=AdamW
			max_words=32
			max_frames=12
			cluster_inter=0
			;;
		04 )
			do_train=0
			do_eval=1
			lr=5e-3
			optim=AdamW
			cluster_inter=1
			cluster_algo='kmediods++'
			minkowski_norm_p=2.0
			cluster_num_blocks='49 49 49 49 49 49 49 49 49 49 49 49'
			target_frames_blocks='12 12 12 12 12 12 6 6 6 6 6 6'
			# batch_size_val=1
			resume=${HOME}/models/eclip/eclip_new_abla_${dataset}_${num}/ckpt.best.pth.tar
			# save_feature_path=${HOME}/output/lsmdc_id
			;;
		09 )
			do_train=0
			do_eval=1
			lr=5e-3
			optim=AdamW
			cluster_inter=1
			cluster_algo='kmediods++'
			minkowski_norm_p=2.0
			cluster_num_blocks='49 49 49 49 49 49 49 49 49 49 49 49'
			target_frames_blocks='12 12 12 12 12 12 4 4 4 4 4 4'
			# batch_size_val=1
			resume=${HOME}/models/eclip/eclip_new_abla_${dataset}_${num}/ckpt.best.pth.tar
			# save_feature_path=${HOME}/output/lsmdc_id_09
			;;
		12 )
			lr=5e-3
			optim=AdamW
			cluster_inter=1
			cluster_algo='kmediods++'
			minkowski_norm_p=2.0
			cluster_num_blocks='49 49 49 49 49 49 49 49 49 49 49 49'
			target_frames_blocks='12 12 12 12 12 12 3 3 3 3 3 3'
			;;
		22 )
			do_train=0
			do_eval=1
			resume=${HOME}/models/eclip/eclip_new_abla_${dataset}_${num}/ckpt.best.pth.tar
			lr=5e-3
			optim=AdamW
			cluster_inter=1
			cluster_algo='spectral'
			minkowski_norm_p=2.0
			spectral_graph='KNN'
			spectral_sigma=2.0
			cluster_num_blocks='49 49 49 49 49 49 49 49 49 49 49 49'
			target_frames_blocks='12 12 12 12 12 12 6 6 6 6 6 6'
			;;
		23 )
			do_train=0
			do_eval=1
			resume=${HOME}/models/eclip/eclip_new_abla_${dataset}_${num}/ckpt.best.pth.tar
			lr=5e-3
			optim=AdamW
			cluster_inter=1
			cluster_algo='spectral'
			minkowski_norm_p=2.0
			spectral_graph='KNN'
			spectral_sigma=2.0
			cluster_num_blocks='49 49 49 49 49 49 49 49 49 49 49 49'
			target_frames_blocks='12 12 12 12 12 12 4 4 4 4 4 4'
			;;
		28 )
			lr=5e-3
			optim=AdamW
			cluster_inter=1
			cluster_algo='kmediods++'
			minkowski_norm_p=2.0
			cluster_num_blocks='49 49 49 49 49 49 49 49 49 49 49 49'
			target_frames_blocks='12 12 12 12 12 12 6 6 6 6 6 6'
			pre_norm=1
			;;
		29 )
			lr=5e-3
			optim=AdamW
			cluster_inter=1
			cluster_algo='kmediods++'
			minkowski_norm_p=2.0
			cluster_num_blocks='49 49 49 49 49 49 49 49 49 49 49 49'
			target_frames_blocks='12 12 12 12 12 12 4 4 4 4 4 4'
			pre_norm=1
			;;
	  * )
		;;
	esac

model_dir=${HOME}/models/eclip/eclip_new_abla_${dataset}_${num}
echo "The model dir is ${model_dir}"

# CUDA_LAUNCH_BLOCKING=1
python ../main.py \
		--do_train ${do_train} \
		--do_eval ${do_eval} \
		--num_thread_reader ${num_workers} \
		--epochs ${epochs} \
		--batch_size ${batch_size} \
		--n_display ${n_display} \
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
		--batch_size_val ${batch_size_val} \
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
		--resume ${resume} \
		--load_from_pretrained ${load_from_pretrained} \
		--save_feature_path ${save_feature_path} \
		--pre_norm ${pre_norm}

done

echo "Training Finished!!!"
