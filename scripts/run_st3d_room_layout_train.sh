# input data folder and output folder
RAW_DATASET_ROOT=$1
OUTPUT_FOLDER=$2

# bedroom parametrization
MODEL_FLAGS="--layout_channels 32 --layout_size 23 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True  --b_class_cond False --b_text_cond True --use_input_encoding True"
MAX_STEPS=400000


DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --schedule_sampler loss-second-moment --use_3d_iou False "
NUM_GPUS=1

train_stats_file=$RAW_DATASET_ROOT/train/bedroom/train_dataset_stats.json

mpiexec -n $NUM_GPUS python src/st3d_room_layout_train.py \
    --data_dir $RAW_DATASET_ROOT/train/bedroom/ \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS \
    --lr_anneal_steps $MAX_STEPS \
    --log_dir $OUTPUT_FOLDER  \
     --dataset_stats_file $train_stats_file 
