MODEL_FLAGS="--layout_channels 33 --layout_size 21 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --schedule_sampler loss-second-moment --use_3d_iou False "
NUM_GPUS=2
ROOM_TYPE="diningroom"
CONFIG_FILE="/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/config/3dfront_diningroom_config.yaml"
MAX_STEPS=400000

mpiexec -n $NUM_GPUS python scripts/threed_front_room_layout_train.py \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS \
    --room_type $ROOM_TYPE  \
    --config_file $CONFIG_FILE \
    --lr_anneal_steps $MAX_STEPS \
    --log_dir "log/3dfront_diningroom/textconditional-0122/"


MODEL_FLAGS="--layout_channels 33 --layout_size 21 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
CONFIG_FILE="/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/config/3dfront_diningroom_config.yaml"

python scripts/threed_front_room_layout_sample.py \
    --model_path log/3dfront_diningroom/textconditional-0122/ema_0.9999_400000.pt \
     $MODEL_FLAGS \
     $DIFFUSION_FLAGS \
     --num_samples 1000 \
      --room_type $ROOM_TYPE \
      --path_to_pickled_3d_futute_models /mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/threed_future_model_diningroom.pkl \
      --config_file $CONFIG_FILE \
      --log_dir "log/3dfront_diningroom/textconditional-0122/sample_results"
