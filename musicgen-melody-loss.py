import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read
from audiocraft.modules.conditioners import ConditioningAttributes

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


model = MusicGen.get_pretrained(
    'facebook/musicgen-melody', device=device)

model.lm = model.lm.to(dtype=torch.float32)
model.compression_model = model.compression_model.to(dtype=torch.float32)

# ================== 音频处理 ==================
audio_path = '/content/drive/MyDrive/MusicGen-Melody-Loss/AudioTest/0.wav'
audio_tensor, sr = audio_read(audio_path)

# 数据预处理（重要：保持float32和设备一致）
audio_tensor = audio_tensor.unsqueeze(0)  # [batch=1, channels, time]
audio_tensor = audio_tensor.to(device).to(torch.float32)

print(f"[验证] 音频张量 | 类型: {audio_tensor.dtype} | 设备: {audio_tensor.device}")

# ================== 编码处理 ==================
with torch.no_grad():
    tokens = model.compression_model.encode(audio_tensor)[0]
    tokens = tokens.long()  # 强制转换为int64（关键！）

print(f"[验证] 编码tokens | 类型: {tokens.dtype} | 设备: {tokens.device}")

# ================== 准备训练数据 ==================
input_tokens = tokens[:, :, :-1].contiguous()  
target_tokens = tokens[:, :, 1:].contiguous() 

print(f"[验证] 输入tokens形状: {input_tokens.shape} | 目标tokens形状: {target_tokens.shape}")

# ================== 损失计算 ==================
conditions = [
    ConditioningAttributes(
        text={'description': "piano music"},
        wav={'self_wav': (audio_tensor, torch.tensor([audio_tensor.shape[-1]], device=device), [sr], [], [])} 
    )
]
print(f"采样率类型: {type(conditions[0].wav['self_wav'][2])}")  # 应输出 <class 'list'>

with torch.no_grad():
    # 禁用自动混合精度（关键设置！）
    with torch.autocast(device_type=device.split(':')[0], enabled=False):
        outputs = model.lm(input_tokens, conditions)
    
    logits = outputs.to(torch.float32).reshape(-1, outputs.shape[-1])
    target = target_tokens.reshape(-1)
    print(f"[最终检查] Logits类型: {logits.dtype} | Target类型: {target.dtype}")
    
    loss = torch.nn.functional.cross_entropy(logits, target)
    print(f"负对数似然损失: {loss.item():.4f}")

# ================== 清理显存 ==================
if device == "cuda":
    torch.cuda.empty_cache()
