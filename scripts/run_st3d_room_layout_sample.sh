# NUM_SAMPLES=30

# # bedroom parametrization
# MODEL_FLAGS="--layout_channels 32 --layout_size 23 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding False"

# # living room parametrization
# # MODEL_FLAGS="--layout_channels 34 --layout_size 44 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding True"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing 250"


# python scripts/st3d_room_layout_sample.py --data_dir /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/text2pano/test/bedroom/ \
#  --model_path log/ST3D_bedroom_textcondition_openai-2023-09-08-15-04-50-375770/ema_0.9999_180000.pt \
#  $MODEL_FLAGS \
#  $DIFFUSION_FLAGS \
#  --room_type 'bedroom' \
#  --num_samples $NUM_SAMPLES

# MODEL_FLAGS="--layout_channels 28 --layout_size 34 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
# NUM_SAMPLES=10
# ROOM_TYPE="kitchen"
# train_stats_file="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240215_text2pano/train/kitchen/train_dataset_stats.json"

# python scripts/st3d_room_layout_sample.py \
#  --data_dir /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240215_text2pano/train/kitchen/ \
#  --model_path log/ST3D_kitchen/2024021_normalized/ema_0.9999_300000.pt \
#  $MODEL_FLAGS \
#  $DIFFUSION_FLAGS \
#  --room_type $ROOM_TYPE \
#  --num_samples $NUM_SAMPLES \
#  --log_dir "log/ST3D_kitchen/2024021_normalized/sample_results" \
#  --dataset_stats_file $train_stats_file 

# MODEL_FLAGS="--layout_channels 34 --layout_size 44 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
# NUM_SAMPLES=40
# ROOM_TYPE="livingroom"
# train_stats_file="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train/livingroom/train_dataset_stats.json"

# python scripts/st3d_room_layout_sample.py \
#  --data_dir /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/test/livingroom/ \
#  --model_path log/ST3D_livingroom/20240223_normalized/ema_0.9999_400000.pt \
#  $MODEL_FLAGS \
#  $DIFFUSION_FLAGS \
#  --room_type $ROOM_TYPE \
#  --num_samples $NUM_SAMPLES \
#  --log_dir "log/ST3D_livingroom/20240223_normalized/sample_results/2024eccv_experiment" \
#  --dataset_stats_file $train_stats_file 

MODEL_FLAGS="--layout_channels 32 --layout_size 23 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
NUM_SAMPLES=0
ROOM_TYPE="bedroom"
train_stats_file="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train/bedroom/train_dataset_stats.json"

python scripts/st3d_room_layout_sample.py \
 --data_dir /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/test/bedroom/ \
 --model_path log/ST3D_bedroom/20240221_gpt4caption/ema_0.9999_300000.pt \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 --room_type $ROOM_TYPE \
 --num_samples $NUM_SAMPLES \
 --log_dir "log/ST3D_bedroom/20240221_gpt4caption/sample_results" \
 --dataset_stats_file $train_stats_file 

# MODEL_FLAGS="--layout_channels 28 --layout_size 34 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
# NUM_SAMPLES=20
# ROOM_TYPE="study"
# USE_GPT_TEXT_DESCRIPTION=False
# train_stats_file="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train/study/train_dataset_stats.json"
# python scripts/st3d_room_layout_sample.py \
#  --data_dir /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/test/study/ \
#  --model_path log/ST3D_study/20240306_physical_loss_alabtion/ema_0.9999_400000.pt \
#  $MODEL_FLAGS \
#  $DIFFUSION_FLAGS \
#  --room_type $ROOM_TYPE \
#  --num_samples $NUM_SAMPLES \
#  --log_dir "log/ST3D_study/20240306_physical_loss_alabtion/w_sample_results" \
#  --dataset_stats_file $train_stats_file \
#  --use_gpt_text_desc $USE_GPT_TEXT_DESCRIPTION

# MODEL_FLAGS="--layout_channels 28 --layout_size 34 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
# NUM_SAMPLES=10
# ROOM_TYPE="bathroom"
# USE_GPT_TEXT_DESCRIPTION=False
# train_stats_file="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train/study/train_dataset_stats.json"

# python scripts/st3d_room_layout_sample.py \
#  --data_dir /mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/test/bathroom/ \
#  --model_path log/ST3D_bathroom/20240309_normalizaed/ema_0.9999_400000.pt \
#  $MODEL_FLAGS \
#  $DIFFUSION_FLAGS \
#  --room_type $ROOM_TYPE \
#  --num_samples $NUM_SAMPLES \
#  --log_dir "log/ST3D_bathroom/20240309_normalizaed/sample_results" \
#  --dataset_stats_file $train_stats_file \
#  --use_gpt_text_desc $USE_GPT_TEXT_DESCRIPTION