#!/bin/bash
#SBATCH --job-name=GMNet_release
#SBATCH --error=GMNet_release.%j.err
#SBATCH --output=GMNet_release.%j.out
#SBATCH --partition=allgroups
#SBATCH --mail-user umberto.michieli1@gmail.com
#SBATCH --mail-type ALL
#SBATCH --gres=gpu:titan_rtx:1
#SBATCH --ntasks=3
#SBATCH --mem=30G
#SBATCH --time=500:00:00

export SINGULARITY_TMPDIR=/nfsd/naslttm3/michieli/tmp/

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/../../../Deeplab


START=`/bin/date +%s`  ### fake start time, for initialization purposes
while [ $(( $(/bin/date +%s) - 5000 )) -lt $START ]; do  ## 1800 seconds (if more than this then sim has started)
    START=`/bin/date +%s`
	
	rm -rf TRAIN_58_parts/
	
	singularity exec --nv tf_113_gpu.sif \
	python -u train.py \
	--logtostderr \
	--train_split train \
	--model_variant resnet_v1_101_beta \
	--atrous_rates 6 \
	--atrous_rates 12 \
	--atrous_rates 18 \
	--output_stride 16 \
	--decoder_output_stride 4 \
	--train_crop_size "513,513" \
	--base_learning_rate 0.005 \
	--train_batch_size 10 \
	--training_number_of_steps 50000 \
	--dataset pascal_voc_seg_58_parts \
	--tf_initial_checkpoint pretrained/imagenet/resnet_v1_101_2018_05_04/model.ckpt \
	--train_logdir TRAIN_58_parts \
	--dataset_dir tfrecord_58parts_complete \
	--model_softmax 1 \
	--graph_loss 1 \
	--weighted_graph 1 \
	--class_dilation_kernel 2 \
	--graph_lambda_loss 0.1 \
	--graph_loss_type 'mean_squared_error' \
	--aux_loss 1 \
	--aux_lambda_loss 0.0001 \
	1> TRAIN_58_parts.txt 2>&1
done



singularity exec --nv tf_113_gpu.sif \
python -u eval.py \
--logtostderr \
--eval_split val \
--model_variant resnet_v1_101_beta \
--atrous_rates 6 \
--atrous_rates 12 \
--atrous_rates 18 \
--output_stride 16 \
--decoder_output_stride 4 \
--eval_crop_size "513,513" \
--checkpoint_dir TRAIN_58_parts \
--eval_logdir EVAL_58_parts \
--dataset pascal_voc_seg_58_parts \
--dataset_dir tfrecord_58parts_complete \
--max_number_of_evaluations 1 \
--model_softmax 1 \
1> EVAL_58_parts.txt 2>&1

