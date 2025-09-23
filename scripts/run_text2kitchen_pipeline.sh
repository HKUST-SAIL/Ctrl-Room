CUDA_VISIBLE_DEVICES=0
NUM_SAMPLES=1

# input data folder and output folder
RAW_DATASET_ROOT=$1
OUTPUT_FOLDER=$2

# run layout sampling
MODEL_FLAGS="--layout_channels 28 --layout_size 34 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
ROOM_TYPE="kitchen"
RAW_DATASET_PATH=$RAW_DATASET_ROOT/test/kitchen/
train_stats_file=$RAW_DATASET_ROOT/train/kitchen/train_dataset_stats.json

LAYOUT_MODEL_PATH=ckpts/st3d_layout_kitchen.pt
python src/st3d_room_layout_sample.py \
 --data_dir $RAW_DATASET_PATH \
 --model_path $LAYOUT_MODEL_PATH \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 --room_type $ROOM_TYPE \
 --num_samples $NUM_SAMPLES \
 --log_dir $OUTPUT_FOLDER \
 --dataset_stats_file $train_stats_file 


# run panorama sampling
PANO_INPUT_FOLDER=$OUTPUT_FOLDER/$ROOM_TYPE
PANOGEN_MODEL_PATH="ckpts/st3d_panorama_kitchen.ckpt"
python src/st3d_panorama_sample.py \
    --input_folder $PANO_INPUT_FOLDER \
    --ckpt_filepath $PANOGEN_MODEL_PATH

# # run panoramic reconstrcution
# python src/st3d_panorama_recons.py \
#     --input_folder $PANO_INPUT_FOLDER \
#     --use_egformer True




