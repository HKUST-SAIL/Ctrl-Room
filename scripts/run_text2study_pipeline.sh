export CUDA_VISIBLE_DEVICES=1
NUM_SAMPLES=-1
OUTPUT_FOLDER=/mnt/nas_3dv/hdd1/fangchuan/eccv2024_ctrlroom/rebuttal/layout_eval_study

eval "$(conda shell.bash hook)"
conda activate structured3d

# run layout sampling
cd /mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/
MODEL_FLAGS="--layout_channels 28 --layout_size 34 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond False --b_text_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing ddim200 --use_ddim True"
ROOM_TYPE="study"
RAW_DATASET_PATH=/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/test/study/
train_stats_file="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train/study/train_dataset_stats.json"
USE_GPT_TEXT_DESCRIPTION=False

LAYOUT_MODEL_PATH=log/ST3D_study/20240229_normalized/ema_0.9999_400000.pt
python scripts/st3d_room_layout_sample.py \
 --data_dir $RAW_DATASET_PATH \
 --model_path $LAYOUT_MODEL_PATH \
 $MODEL_FLAGS \
 $DIFFUSION_FLAGS \
 --room_type $ROOM_TYPE \
 --num_samples $NUM_SAMPLES \
 --log_dir $OUTPUT_FOLDER \
 --dataset_stats_file $train_stats_file \
 --use_gpt_text_desc $USE_GPT_TEXT_DESCRIPTION



# # run panorama sampling
conda activate control-v11
PANO_INPUT_FOLDER=$OUTPUT_FOLDER/$ROOM_TYPE
APP_MODEL_PATH="/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/ckpts/control_v11p_sd15_seg_study_fullres_22000.ckpt"
cd /mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts
python st3d_panorama_sample.py --input_folder $PANO_INPUT_FOLDER --ckpt_filepath $APP_MODEL_PATH

# # run super-resolution
# python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
#                                                         --ckpt ../stablesr_000117.ckpt \
#                                                         --vqgan_ckpt ../vqgan_cfw_00011.ckpt \
#                                                         --init-img input_img_1 \
#                                                         --outdir output_img_1 \
#                                                         --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
# run panoramic reconstrcution
python st3d_panorama_recons.py --input_folder $PANO_INPUT_FOLDER --use_egformer True




