import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read
from audiocraft.modules.conditioners import ConditioningAttributes

model = MusicGen.get_pretrained('facebook/musicgen-melody')

audio_tensor, sr = audio_read('/path/to/0.wav')
audio_tensor = audio_tensor.unsqueeze(0)

with torch.no_grad():
    tokens,_ = model.compression_model.encode(audio_tensor)

conditions = [ConditioningAttributes(text={'description': ""})]

input_tokens = tokens[:, :, :-1]  
target_tokens = tokens[:, :, 1:]  

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


