# import matplotlib
import matplotlib.pylab as plt

import numpy as np
import torch
from scipy.io.wavfile import write
import os

from hparams import create_hparams
# from model import Tacotron2
# from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from text import text_to_sequence

import sys
sys.path.append('waveglow')
from denoiser import Denoiser


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')
    plt.show()


def save_audio(path, sampling_rate, audio):
    print("saving audio to", path)
    write(path, sampling_rate, audio.astype(np.float32))


def load_waveglow(path):
    waveglow = torch.load(path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    return waveglow, denoiser


def infer(tacotron2, waveglow_path, text, audio_path):
    hparams = create_hparams()
    hparams.max_wav_value=32768.0
    hparams.sampling_rate = 22050
    hparams.filter_length=1024
    hparams.hop_length=256
    hparams.win_length=1024
    waveglow, denoiser = load_waveglow(waveglow_path)

    sequence = np.array(text_to_sequence(text, ['german_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    # text -> mel spectogram
    mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.inference(sequence)

    # mel spectogram -> sound wave
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.85)

    # denoise
    audio_denoised = denoiser(audio, strength=0.006)[:, 0]
    
    audio_denoised_np = audio_denoised.cpu().numpy()    
    
    if audio_path:
        save_audio(audio_path, hparams.sampling_rate, audio_denoised_np)

    return audio_denoised_np
