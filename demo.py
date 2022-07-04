import matplotlib
import matplotlib.pylab as plt

import numpy as np
import torch
from scipy.io.wavfile import write
import os

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')
    plt.show()


def save_audio(out_dir, file_name, sampling_rate, audio):
    audio_path = os.path.join(out_dir, "{}_synthesis.wav".format(file_name))
    print("saving audio to", audio_path)
    write(audio_path, sampling_rate, audio.astype(np.float32))


def load_tacotron2(path, hparams):
    print("load tacotron from", path)
    model = load_model(hparams)
    model.load_state_dict(torch.load(path)['state_dict'])
    model.cuda().eval().half()

    return model


def load_waveglow(path):
    print("load waveglow from", path)
    waveglow = torch.load(path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    return waveglow, denoiser


def infer(tacotron2_path, waveglow_path, text, save=False):
    hparams = create_hparams()
    hparams.max_wav_value=32768.0
    hparams.sampling_rate = 22050
    hparams.filter_length=1024
    hparams.hop_length=256
    hparams.win_length=1024
    tacotron2 = load_tacotron2(tacotron2_path, hparams)
    waveglow, denoiser = load_waveglow(waveglow_path)

    sequence = np.array(text_to_sequence(text, ['german_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    # text -> mel spectogram
    print("synthesizing", text)
    mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.inference(sequence)

    # mel spectogram -> sound wave
    print("generating audio", text)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.85)

    # denoise
    audio_denoised = denoiser(audio, strength=0.006)[:, 0]
    
    audio_denoised_np = audio_denoised.cpu().numpy()    
    
    if save:
        save_audio('foo', 'bar', hparams.sampling_rate, audio_denoised_np)

    return audio_denoised_np