export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/blob/v-yuancwang/DiffAudioImg/AudioControlNet_lora"

accelerate launch train/train_controlnet_lora.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --resolution=512 \
 --learning_rate=2e-5 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=1 \
 --max_train_steps=50000 \
 --checkpointing_steps=2000 \
 --image_column="image" \
 --conditioning_image_column="conditioning_mel" \
 --caption_column="text" \
 --proportion_empty_prompts=0.8 \
 --tracker_project_name="train_controlnet_lora" \