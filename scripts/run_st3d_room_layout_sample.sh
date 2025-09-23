# sample living room
MODEL_FLAGS="--layout_channels 34 --layout_size 44 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
NUM_SAMPLES=1
ROOM_TYPE="livingroom"
train_stats_file="/project/lrmcongen/data/Structured3D/20240219_text2pano/train/livingroom/train_dataset_stats.json"

python src/st3d_room_layout_sample.py \
 --data_dir /project/lrmcongen/data/Structured3D/20240219_text2pano/test/livingroom/ \
 --model_path ckpts/st3d_layout_livingroom.pt \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 --room_type $ROOM_TYPE \
 --num_samples $NUM_SAMPLES \
 --log_dir "sample_results" \
 --dataset_stats_file $train_stats_file 

# sample bedroom
MODEL_FLAGS="--layout_channels 32 --layout_size 23 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
NUM_SAMPLES=1
ROOM_TYPE="bedroom"
train_stats_file="/project/lrmcongen/data/Structured3D/20240219_text2pano/train/bedroom/train_dataset_stats.json"

python src/st3d_room_layout_sample.py \
 --data_dir /project/lrmcongen/data/Structured3D/20240219_text2pano/test/bedroom/ \
 --model_path ckpts/st3d_layout_bedroom.pt \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 --room_type $ROOM_TYPE \
 --num_samples $NUM_SAMPLES \
 --log_dir "sample_results" \
 --dataset_stats_file $train_stats_file 

# sample study
MODEL_FLAGS="--layout_channels 28 --layout_size 34 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
NUM_SAMPLES=1
ROOM_TYPE="study"
train_stats_file="/project/lrmcongen/data/Structured3D/20240219_text2pano/train/study/train_dataset_stats.json"
python src/st3d_room_layout_sample.py \
 --data_dir /project/lrmcongen/data/Structured3D/20240219_text2pano/test/study/ \
 --model_path ckpts/st3d_layout_study.pt \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 --room_type $ROOM_TYPE \
 --num_samples $NUM_SAMPLES \
 --log_dir "sample_results" \
 --dataset_stats_file $train_stats_file 

# sample bathroom
# MODEL_FLAGS="--layout_channels 28 --layout_size 34 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
# NUM_SAMPLES=1
# ROOM_TYPE="bathroom"
# train_stats_file="/project/lrmcongen/data/Structured3D/20240219_text2pano/train/study/train_dataset_stats.json"

# python src/st3d_room_layout_sample.py \
#  --data_dir /project/lrmcongen/data/Structured3D/20240219_text2pano/test/bathroom/ \
#  --model_path ckpts/st3d_layout_bathrooom.pt \
#  $MODEL_FLAGS \
#  $DIFFUSION_FLAGS \
#  --room_type $ROOM_TYPE \
#  --num_samples $NUM_SAMPLES \
#  --log_dir "sample_results" \
#  --dataset_stats_file $train_stats_file 
