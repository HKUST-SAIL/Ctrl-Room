MODEL_FLAGS="--layout_channels 34 --layout_size 44 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --schedule_sampler loss-second-moment --use_3d_iou False "
NUM_GPUS=2
MAX_STEPS=400000
train_stats_file="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train/livingroom/train_dataset_stats.json"
USE_GPT_TEXT_DESCRIPTION=True

mpiexec -n $NUM_GPUS python scripts/st3d_room_layout_train.py \
 --data_dir /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train/livingroom/ \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 $TRAIN_FLAGS \
 --lr_anneal_steps $MAX_STEPS \
 --log_dir log/ST3D_livingroom/20240227_gpt_caption \
 --dataset_stats_file $train_stats_file \
 --use_gpt_text_desc $USE_GPT_TEXT_DESCRIPTION



MODEL_FLAGS="--layout_channels 34 --layout_size 44 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
NUM_SAMPLES=10
ROOM_TYPE="livingroom"
train_stats_file="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train/livingroom/train_dataset_stats.json"

python scripts/st3d_room_layout_sample.py \
 --data_dir /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/test/livingroom/ \
 --model_path log/ST3D_livingroom/20240227_gpt_caption/ema_0.9999_400000.pt \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 --room_type $ROOM_TYPE \
 --num_samples $NUM_SAMPLES \
 --log_dir "log/ST3D_livingroom/20240227_gpt_caption/sample_results" \
 --dataset_stats_file $train_stats_file \
 --use_gpt_text_desc $USE_GPT_TEXT_DESCRIPTION

