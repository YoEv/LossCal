import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.compression_model.eval()
model.lm.eval()
processor = model.processor()

waveform, sr = audio_read('/Volumes/MCastile/bolero_ravel.mp3')
inputs = processor(
    audio=waveform,
    sampling_rate=sr,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model(**inputs, lable=inputs["input_ids"])
    loss = outputs.loss
print(f"loss: {loss.item()}")
##