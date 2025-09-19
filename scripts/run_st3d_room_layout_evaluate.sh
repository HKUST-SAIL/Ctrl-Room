MODEL_FLAGS="--layout_channels 23 --layout_size 32 --num_channels 128 --num_res_blocks 3 --b_learn_sigma True --b_class_cond True --use_input_encoding True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine  --timestep_respacing 250"


python evaluate_object_category_kl_divergency.py --model_path log/openai-2023-07-07-19-13-25-077423/ema_0.9999_240000.pt $MODEL_FLAGS $DIFFUSION_FLAGS