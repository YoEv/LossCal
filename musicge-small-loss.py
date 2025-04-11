import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read
from audiocraft.modules.conditioners import ConditioningAttributes

model = MusicGen.get_pretrained('facebook/musicgen-small')

audio_tensor, sr = audio_read('/Volumes/MCastile/Castile/HackerProj/Audiocraft_Demo/0.wav')
audio_tensor = audio_tensor.unsqueeze(0)

with torch.no_grad():
    tokens,_ = model.compression_model.encode(audio_tensor)

conditions = [ConditioningAttributes(text={'description': ""})]

input_tokens = tokens[:, :, :-1]  
print(f"input_tokens.shape: {input_tokens.shape}")
target_tokens = tokens[:, :, 1:]  
print(f"target_tokens.shape: {target_tokens.shape}")

with torch.no_grad():
    outputs = model.lm(
    input_tokens, 
    conditions) 

    B, K, T = input_tokens.shape
    logits = outputs.reshape(-1, outputs.shape[-1])
    target = target_tokens.reshape(-1)
    print(f"logits.shape: {logits.shape}")

    loss = torch.nn.functional.cross_entropy(logits,target)
    print(f"Negative Log-Likelihood Loss: {loss.item()}")


# input_tokens.shape: torch.Size([1, 4, 399])
# target_tokens.shape: torch.Size([1, 4, 399])
# logits.shape: torch.Size([1596, 2048])
# Negative Log-Likelihood Loss: 5.701778411865234