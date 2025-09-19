# run text2light pipeline



OUTPUT_FOLDER=/mnt/nas_3dv/hdd1/fangchuan/mesh_generation_experiments/text2light_results/

eval "$(conda shell.bash hook)"

# generate HDR panorama
# conda activate text2light
# cd /mnt/nas_3dv/hdd1/fangchuan/Text2Light
# bash test_text.sh

# run MVDiffusion 
OUTPUT_FOLDER=/mnt/nas_3dv/hdd1/fangchuan/mesh_generation_experiments/2024eccv_experiments/text2light_study/
# conda activate mvdiffusion
# python test_text2room.py --out_dir $OUTPUT_FOLDER

# run panorama sampling
conda activate control-v11
PANO_INPUT_FOLDER=$OUTPUT_FOLDER
cd /mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts

# # run super-resolution
# python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
#                                                         --ckpt ../stablesr_000117.ckpt \
#                                                         --vqgan_ckpt ../vqgan_cfw_00011.ckpt \
#                                                         --init-img input_img_1 \
#                                                         --outdir output_img_1 \
#                                                         --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
# run panoramic reconstrcution
python st3d_panorama_recons.py --input_folder $OUTPUT_FOLDER


