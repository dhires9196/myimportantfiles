#Wav2Vec2 is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
##managing audio files as it needs to sample to 16000 HZ
import librosa
#it is from favebook ai
import torch
#Wav2Vec2 model was trained using connectionist temporal classification (CTC) 
#so the model output has to be decoded using Wav2Vec2CTCTokenizer.
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

#load pre-trained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

#load any audio file of your choice
speech, rate = librosa.load("batman1.wav",sr=16000)

import IPython.display as display
display.Audio("batman1.wav", autoplay=True)