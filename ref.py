import os
import json
import pickle
import time
import math
import numpy as np
import torch
from torch import nn, Tensor
import torchaudio
from torchmetrics.text import WordErrorRate
from einops import rearrange, repeat
import torchaudio.transforms as T
from torchaudio import functional as F
from typing import Optional 
from seamless_communication.models.inference import Translator
from optimizer_inhouse import get_optimizer
from voicebox_inhouse import (
    VoiceBox,
    ConditionalFlowMatcherWrapper
)

EPOCHS = 75000
print(torch.__version__)
clip_val = 0.2
seed = 0


# torch.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"With {torch.cuda.device_count()} GPUs")
torch.autograd.set_detect_anomaly(True)

transcript_file = '/raid/ganesh/ishapandey/isha/Koustubh/mnt/indic_tts/train/transcript.txt'
hindi_wav_data = '/raid/ganesh/ishapandey/isha/Koustubh/mnt/indic_tts/train/'
aligned_data = '/raid/ganesh/ishapandey/isha/Koustubh/mnt/aligned_data_indic_tts_hindi_fem_spkr1/'

labels = open('/raid/ganesh/ishapandey/isha/Koustubh/voicebox/config/hindi/dict.ltr.json', 'r')
labels = json.load(labels)
labels = ['<s>', '<pad>', '</s>', '<unk>'] + [i[0] for i in labels.items()]

NUM_PHONEMES = len(labels)
dictionary = {c: i for i, c in enumerate(labels)}

# default values used; refer https://github.com/lucidrains/voicebox-pytorch/blob/main/voicebox_pytorch/voicebox_pytorch.py
# a lot more hyperparameters exists, check and link and fine tune

NEW = False
PATH = './stored_models/chk_16f'
AUDIO_DIM = 80
DEPTH = 16 # OLD 24
DIM_HEAD = 64
HEADS = 16
BATCH_SIZE = 50
MAX_LEN = 1600
WARM_UP = 50

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.random.manual_seed(seed)


model = VoiceBox(
    dim_in = AUDIO_DIM,
    num_cond_tokens = NUM_PHONEMES,
    depth = DEPTH,
    dim_head = DIM_HEAD,
    heads = HEADS
)

optimizer = get_optimizer(model.parameters(), lr = 1e-4)
step = 0

if not NEW and os.path.exists(PATH):
    # model = torch.load(PATH)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']

cfm_wrapper = ConditionalFlowMatcherWrapper(
    voicebox = model,
    use_torchode = False   # by default will use torchdiffeq with midpoint as in paper, but can use the promising torchode package too
)




class InverseMelScale(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        norm: Optional[str] = None,
        mel_scale: str = 'htk'
    ) -> None: 
        super().__init__()


        f_max = f_max or float(sample_rate // 2)
        fb = F.melscale_fbanks(
            (n_fft // 2 + 1), f_min, f_max, n_mels, sample_rate, norm, mel_scale
        )
        self.register_buffer("fb", torch.linalg.pinv(fb))

    def forward(self, melspec: Tensor) -> Tensor:
        # Flatten the melspec except for the frequency and time dimension
        shape = melspec.shape
        melspec = rearrange(melspec, "... f t -> (...) f t")
        fb = repeat(self.fb, "f m -> n m f", n=melspec.shape[0])
        specgram = fb @ melspec
        specgram = torch.clamp(specgram, min=0.)
        specgram = specgram.view(shape[:-2] + (fb.shape[-2], shape[-1]))

        return specgram

class dB_to_Amplitude(nn.Module):
    def __call__(self, features):
        return(torch.from_numpy(np.power(10.0, features.numpy()/10.0)))

class MelVoco():
    def __init__(
        self,
        *,
        log = True,
        n_mels = 80,
        sampling_rate = 16000, # old, 24000
        f_max = 8000,
        n_fft = 1024,
        win_length = 640,
        hop_length = 160
    ):
        self.log = log
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_max = f_max
        self.win_length = win_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate


    def encode(self, audio):
        stft_transform = T.Spectrogram(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            window_fn = torch.hann_window
        )

        spectrogram = stft_transform(audio)

        mel_transform = T.MelScale(
            n_mels = self.n_mels,
            sample_rate = self.sampling_rate,
            n_stft = self.n_fft // 2 + 1,
            f_max = self.f_max
        )

        mel = mel_transform(spectrogram)

        if self.log:
            mel = T.AmplitudeToDB()(mel)

        mel = rearrange(mel, 'b d n -> b n d')
        return mel

    def decode(self, mel):
        mel = rearrange(mel, 'b n d -> b d n')
        inverse_transform = torch.nn.Sequential(
            dB_to_Amplitude(), 
            InverseMelScale(sample_rate = self.sampling_rate, n_fft = self.n_fft, n_mels = self.n_mels, f_max = self.f_max),
            T.GriffinLim(n_fft = self.n_fft, hop_length = self.hop_length, win_length = self.win_length, window_fn = torch.hann_window))
        waveform = inverse_transform(torch.squeeze(mel))
        return torch.unsqueeze(waveform,0)

def inference_one(file_name):
    local_loc = 'local_16f.wav'
    audio_dir = hindi_wav_data
    alignment_dir = aligned_data
    file_loc = os.path.join(audio_dir, file_name)
    align_loc = os.path.join(alignment_dir, file_name.split('.')[0] + '.pkl')
    f = open(transcript_file, 'r')
    target_sentence = ''
    model.eval()

    for i in f.readlines():
        w = i.split()

        if(len(w) <= 1):
            continue

        if(w[0] == file_name.split('.')[0]):
            target_sentence = ' '.join(w[1:])
            break


    try:
        fd = open(align_loc, 'rb')
        aligned_list = pickle.load(fd)
        fd.close()
    except:
        print('error1')
        return -1


    try:
        waveform, sample_rate = torchaudio.load(file_loc, normalize = True)
    except:
        print('error2')
        return -1

    mels = MelVoco(sampling_rate = sample_rate)
    audio = mels.encode(waveform)

    phoneme_aligner = phoneme_aligner = np.ones((audio.shape[0], audio.shape[1]), dtype = int)
    audio_length = audio.shape[1]

    ex = (audio_length)//2 - aligned_list[-1][3]
    aligned_list = [['<s>', 0.5, 0, aligned_list[0][2]]] + aligned_list + [['</s>', 0.5, aligned_list[-1][3], aligned_list[-1][3] + ex]]

    i = 0
    extra = 0
    
    for phoneme in aligned_list:
        val = ((phoneme[3] - phoneme[2])*audio_length)/(aligned_list[-1][3])
        length = int(val)
        extra = extra + val - length

        if extra >= 1:
            extra = extra - 1
            i = i + 1

        for _ in range(length):
            try:
                phoneme_aligner[0][i] = dictionary[phoneme[0]]
            except:
                if phoneme[0] == ',' or phoneme[0] == '.':
                    phoneme_aligner[0][i] = 1
                else:
                    phoneme_aligner[0][i] = 3


            i += 1

    phoneme_aligner = torch.from_numpy(phoneme_aligner)

    sampled = cfm_wrapper.generate(cond = audio, cond_token_ids = phoneme_aligner)

    # VOCODER CONVERT TO SENTENCE

    wav = mels.decode(sampled)
    torchaudio.save(
        local_loc,
        wav.cpu(),
        sample_rate = sample_rate,
    )

    # SEAMLESSM4T
    in_type = torch.float16
    if device == torch.device('cpu'):
        in_type = torch.float32

    translator = Translator("seamlessM4T_large", "vocoder_36langs", device, in_type)
    transcribed_text, _, _ = translator.predict(local_loc, 'asr', 'hin')
    preds = [transcribed_text.bytes().decode()]

    target = [target_sentence.replace(',', '')]
    wer = WordErrorRate()
    wer_error = wer(preds, target)
    print(preds, target)

    return wer_error



def train(epochs):
    global step
    audio_dict = {}
    start = time.time()
    for file in os.listdir(hindi_wav_data):
        if not file.endswith('.wav'):
            print('Transcript file found')
            continue

        audio_dict[file.split('.')[0]] = os.path.join(hindi_wav_data, file)

    model.train()

    batch_count = 0
    audio_tens = torch.zeros((BATCH_SIZE, MAX_LEN, AUDIO_DIM))
    align_tens = torch.ones((BATCH_SIZE, MAX_LEN), dtype = int)

    for epoch in range(epochs):
        print(f'Starting epoch: {epoch}')
        prin = 0
        remove_list = []
        
        for file in sorted(os.listdir(aligned_data)):
            if file.split('.')[0] in remove_list:
                continue

            # if prin > 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = (param_group['lr']*prin)/(prin+1)


            fd = open(os.path.join(aligned_data, file), 'rb')
            aligned_list = pickle.load(fd)
            fd.close()


            try:
                waveform, sample_rate = torchaudio.load(audio_dict[file.split('.')[0]], normalize = True)
                waveform = torchaudio.functional.resample(waveform, orig_freq = sample_rate, new_freq = 16000)
                sample_rate = 16000
                batch_count += 1
            except:
                continue

            mels = MelVoco(sampling_rate = sample_rate)
            audio = mels.encode(waveform)

            phoneme_aligner = np.ones((audio.shape[0], audio.shape[1]), dtype = int)
            audio_length = audio.shape[1]

            ex = (audio_length)//2 - aligned_list[-1][3]
            aligned_list = [['<s>', 0.5, 0, aligned_list[0][2]]] + aligned_list + [['</s>', 0.5, aligned_list[-1][3], aligned_list[-1][3] + ex]]

            i = 0
            extra = 0
            
            for phoneme in aligned_list:
                val = ((phoneme[3] - phoneme[2])*audio_length)/(aligned_list[-1][3])
                length = int(val)
                extra = extra + val - length

                if extra >= 1:
                    extra = extra - 1
                    i = i + 1

                for _ in range(length):
                    try:
                        phoneme_aligner[0][i] = dictionary[phoneme[0]]
                    except:
                        if phoneme[0] == ',' or phoneme[0] == '.':
                            phoneme_aligner[0][i] = 1
                        else:
                            phoneme_aligner[0][i] = 3


                    i += 1


            phoneme_aligner = torch.from_numpy(phoneme_aligner)

            if audio_length > MAX_LEN:
                audio_tens[batch_count - 1] = audio[0][0:MAX_LEN][:]
                align_tens[batch_count - 1] = phoneme_aligner[0][0:MAX_LEN]

            else:
                audio_tens[batch_count - 1][0:audio_length][:] = audio[0][:][:]
                align_tens[batch_count - 1][0:audio_length] = phoneme_aligner[0][:]

            if batch_count == BATCH_SIZE:
                step += 1

                # WARMING UP AND LINEAR DECAY
                
                # if NEW and step <= WARM_UP:
                #     if step > 1:
                #         for param_group in optimizer.param_groups:
                #             param_group['lr'] = (param_group['lr']*step)/(step-1)
                # else:
                #     for param_group in optimizer.param_groups:
                #         param_group['lr'] = (param_group['lr']*(step - WARM_UP))/(step + 1 - WARM_UP)


                optimizer.zero_grad()   # zero the gradient buffers
                loss = cfm_wrapper(x1 = audio_tens, cond_token_ids = align_tens, cond = audio_tens) # The model employs a better cond_mask [masking parameter]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val) # clipping

                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(name, torch.min(torch.abs(param)), torch.max(torch.abs(param)), torch.isnan(param).any(), torch.min(torch.abs(param.grad)), torch.max(torch.abs(param.grad)), torch.isnan(param.grad).any())
                #     else:
                #         print(name, torch.isnan(param).any(), 'None')

                optimizer.step()    # Does the update
                print(f'Loss at file {prin} = {loss}; time = {time.time() - start}')

                if math.isnan(loss):
                    return -1
                else:
                    # torch.save(model, PATH)

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': step
                        }, PATH)
                    
                
                batch_count = 0
                audio_tens = torch.zeros((BATCH_SIZE, MAX_LEN, AUDIO_DIM))
                align_tens = torch.ones((BATCH_SIZE, MAX_LEN), dtype = int)
                

            prin += 1

    # torch.save(model, PATH)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
        }, PATH)
    
    return 0




train(EPOCHS)


inference_audio = 'train_hindifemale_01286.wav'
wer_error = inference_one(inference_audio)
print(f'The wer error obtained: {wer_error}')
