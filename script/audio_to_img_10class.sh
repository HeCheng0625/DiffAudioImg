export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/blob/v-yuancwang/DiffAudioImg/AudioControlNet_10class"

accelerate launch train/train_controlnet_10class.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --resolution=512 \
 --learning_rate=9.6e-5 \
 --train_batch_size=6 \
 --gradient_accumulation_steps=2 \
 --max_train_steps=500000 \
 --checkpointing_steps=5000 \
 --image_column="image" \
 --conditioning_image_column="conditioning_mel" \
 --caption_column="text" \
 --proportion_empty_prompts=0.8 \
 --tracker_project_name="train_controlnet_10class" \