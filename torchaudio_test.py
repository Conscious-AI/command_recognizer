import torchaudio

wav_tensor, rate = torchaudio.load('./sample/is_the_fox_online/itfo-1.wav')

print(wav_tensor)
print(rate)