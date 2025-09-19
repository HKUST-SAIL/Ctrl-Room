export CUDA_VISIBLE_DEVICES=0
NUM_SAMPLES=-1
OUTPUT_FOLDER=/mnt/nas_3dv/hdd1/fangchuan/eccv2024_ctrlroom/rebuttal/layout_eval_livingroom

eval "$(conda shell.bash hook)"
conda activate structured3d

# # run layout sampling
cd /mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/
MODEL_FLAGS="--layout_channels 34 --layout_size 44 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
ROOM_TYPE="livingroom"
RAW_DATASET_PATH=/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/test/livingroom/
train_stats_file="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train/livingroom/train_dataset_stats.json"

LAYOUT_MODEL_PATH=log/ST3D_livingroom/20240223_normalized/ema_0.9999_400000.pt
python scripts/st3d_room_layout_sample.py \
 --data_dir $RAW_DATASET_PATH \
 --model_path $LAYOUT_MODEL_PATH \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 --room_type $ROOM_TYPE \
 --num_samples $NUM_SAMPLES \
 --log_dir $OUTPUT_FOLDER \
 --dataset_stats_file $train_stats_file 

# run panorama sampling
# PANO_INPUT_FOLDER=$OUTPUT_FOLDER/livingroom
PANO_INPUT_FOLDER=/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/3dv_experiments/bedrooms/
APP_MODEL_PATH="/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/ckpts/control_v11p_sd15_seg_livingroom_fullres_40000.ckpt"
conda activate control-v11
cd /mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts
python st3d_panorama_sample.py --input_folder $PANO_INPUT_FOLDER --ckpt_filepath $APP_MODEL_PATH
# run super-resolution

# run panoramic reconstrcution
python st3d_panorama_recons.py --input_folder $PANO_INPUT_FOLDER --use_egformer True


