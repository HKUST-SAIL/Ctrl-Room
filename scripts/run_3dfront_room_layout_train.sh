# MODEL_FLAGS="--layout_channels 36 --layout_size 41 --attention_resolutions 32,16,8 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding False"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
# TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment --use_3d_iou False "
# NUM_GPUS=2
# ROOM_TYPE="livingroom"

# mpiexec -n $NUM_GPUS python scripts/threed_front_room_layout_train.py \
#     --data_dir /mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/full/train/livingroom/ \
#     $MODEL_FLAGS \
#     $DIFFUSION_FLAGS \
#     $TRAIN_FLAGS \
#     --room_type $ROOM_TYPE 
# mpiexec -n $NUM_GPUS python -m cProfile -o program.prof scripts/threed_front_room_layout_train.py --data_dir /mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/bedroom/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

# MODEL_FLAGS="--layout_channels 30 --layout_size 13 --attention_resolutions 32,16,8 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
# TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment --use_3d_iou False "
# NUM_GPUS=2
# ROOM_TYPE="bedroom"
# CONFIG_FILE="/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/config/3dfront_bedroom_config.yaml"
# MAX_STEPS=200000

# mpiexec -n $NUM_GPUS python scripts/threed_front_room_layout_train.py \
#     --data_dir /mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231112/threed_front_bedroom/ \
#     $MODEL_FLAGS \
#     $DIFFUSION_FLAGS \
#     $TRAIN_FLAGS \
#     --room_type $ROOM_TYPE  \
#     --config_file $CONFIG_FILE \
#     --lr_anneal_steps $MAX_STEPS \
#     --log_dir "log/3dfront_bedroom/"


MODEL_FLAGS="--layout_channels 33 --layout_size 21 --attention_resolutions 32,16,8 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --schedule_sampler loss-second-moment --use_3d_iou False "
NUM_GPUS=2
ROOM_TYPE="livingroom"
CONFIG_FILE="/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/config/3dfront_livingroom_config.yaml"
MAX_STEPS=200000

mpiexec -n $NUM_GPUS python scripts/threed_front_room_layout_train.py \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS \
    --room_type $ROOM_TYPE  \
    --config_file $CONFIG_FILE \
    --lr_anneal_steps $MAX_STEPS \
    --log_dir "log/3dfront_livingroom/"

# MODEL_FLAGS="--layout_channels 33 --layout_size 21 --attention_resolutions 32,16,8 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
# TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --schedule_sampler loss-second-moment --use_3d_iou False "
# NUM_GPUS=2
# ROOM_TYPE="diningroom"
# CONFIG_FILE="/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/config/3dfront_diningroom_config.yaml"
# MAX_STEPS=200000

# mpiexec -n $NUM_GPUS python scripts/threed_front_room_layout_train.py \
#     $MODEL_FLAGS \
#     $DIFFUSION_FLAGS \
#     $TRAIN_FLAGS \
#     --room_type $ROOM_TYPE  \
#     --config_file $CONFIG_FILE \
#     --lr_anneal_steps $MAX_STEPS \
#     --log_dir "log/3dfront_diningroom/"