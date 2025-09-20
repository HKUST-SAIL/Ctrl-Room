export CUDA_VISIBLE_DEVICES=1
NUM_SAMPLES=-1
OUTPUT_FOLDER=sample_results

# run layout sampling
MODEL_FLAGS="--layout_channels 32 --layout_size 23 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
ROOM_TYPE="bedroom"
RAW_DATASET_PATH=/path/to/your/CtrlRoom_dataset/test/bedroom/
train_stats_file="/path/to/your/CtrlRoom_dataset/train/bedroom/train_dataset_stats.json"

LAYOUT_MODEL_PATH=ckpts/bedroom_ema_0.9999_400000.pt
python src/st3d_room_layout_sample.py \
 --data_dir  $RAW_DATASET_PATH\
 --model_path $LAYOUT_MODEL_PATH \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 --room_type $ROOM_TYPE \
 --num_samples $NUM_SAMPLES \
 --log_dir $OUTPUT_FOLDER \
 --dataset_stats_file $train_stats_file 

# # run panorama sampling
PANO_INPUT_FOLDER=$OUTPUT_FOLDER/bedroom
PANOGEN_MODEL_PATH="ckpts/control_v11p_sd15_seg_bedroom_fullres_32000.ckpt"
python src/st3d_panorama_sample.py \
    --input_folder $PANO_INPUT_FOLDER \
    --ckpt_filepath $PANOGEN_MODEL_PATH

# run panoramic reconstrcution
python src/st3d_panorama_recons.py \
    --input_folder $PANO_INPUT_FOLDER \
    --use_egformer True




