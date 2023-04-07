python /home/v-yuancwang/hifi-gan/train.py \
--input_wavs_dir="/blob/v-yuancwang/audio_editing_data/audioset96/wav" \
--input_training_file="/home/v-yuancwang/hifi-gan/audioset96_train.txt" \
--input_validation_file="/home/v-yuancwang/hifi-gan/audioset96_val.txt" \
--checkpoint_path="/blob/v-yuancwang/hifigan_cp" \
--config="/home/v-yuancwang/hifi-gan/config_ours.json" \
--training_epochs=500 \
--stdout_interval=5 \
--checkpoint_interval=5000 \
--summary_interval=100 \
--validation_interval=1000