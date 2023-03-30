from scipy.io.wavfile import read
import torchaudio
import torch
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import os
from tqdm import tqdm
import sys

MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

vgg_audio_path = "/blob/v-yuancwang/DiffAudioImg/VGGSound/data/vggsound/audio"
vgg_video_path = "/blob/v-yuancwang/DiffAudioImg/VGGSound/data/vggsound/video"

vgg_audio_lists = os.listdir(vgg_audio_path)
vgg_video_lists = os.listdir(vgg_video_path)

print(len(vgg_audio_lists))
vgg_audio_lists.sort()
print(vgg_audio_lists[:5])
print(len(vgg_video_lists))
vgg_video_lists.sort()
print(vgg_video_lists[:5])

vgg_wav_path = "/blob/v-yuancwang/DiffAudioImg/VGGSound/data/vggsound/wav"
vgg_mel_path = "/blob/v-yuancwang/DiffAudioImg/VGGSound/data/vggsound/mel"

def flac_to_wav(file_name, save_file_name):
    os.system("/usr/bin/ffmpeg -y -i" + " " + file_name + " " + save_file_name)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

for f in tqdm(vgg_audio_lists[:]):
    file_name = os.path.join(vgg_audio_path, f)
    save_file_name = f[:11]+".wav"
    save_file_name = os.path.join(vgg_wav_path, save_file_name)
    with HiddenPrints():
        flac_to_wav(file_name, save_file_name)

vgg_wav_lists = os.listdir(vgg_wav_path)
print(len(vgg_wav_lists))
vgg_wav_lists.sort()
print(vgg_wav_lists[:5])