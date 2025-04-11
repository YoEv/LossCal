# LossCal

# Testing - 4/9/2025 - Log

### Summery：

In order to figure out whether a generative  model could polish a piece of music or a established verifier could score a piece of music, and to figure out whether I totally understood how to get the loss or likelihood from gen model, I tried Audiobox-Aesthetics, MusicGen-small, and MusicGen-melody to give the generated music a score.  The  Audiobox-Aesthetics give both the ground truth “Wagner Ori” and all other modified or generated music a score in four dimensions. Depending on the dimension meta provided, it seems promising that these dimension could score the music in the way they defined, though it might not satisfied the name “aesthetics”. While I was trying the MusicGen-small and MusicGen-melody, in the Colab both of them work, but locally I achieve the success of only the MusicGen-small due to the computer memory limit. Chun will help with the next testing step, if successes, I will ssh to his server. I haven’t find the direct way of generating the loss from the generated model. Notes left and experiments recorded. Thanks~~~! 

### Tried Audiobox-Aesthetics to scoring the music pieces:

Input files:

Wagner’s Original music, Xiaosha modified music, Logic generated music

```jsx
{"path":"/Volumes/MCastile/Castile/MForward/Garbage to Better Music/DataClean/Bounces/Wagner_RH/RH_Ori.wav"}
{"path":"/Volumes/MCastile/Castile/MForward/Garbage to Better Music/DataClean/Bounces/Wagner_RH/RH_1.wav"}
{"path":"/Volumes/MCastile/Castile/MForward/Garbage to Better Music/DataClean/Bounces/Wagner_RH/RH_2.wav"}
{"path":"/Volumes/MCastile/Castile/MForward/Garbage to Better Music/DataClean/Bounces/Wagner_RH/RH_3.wav"}
{"path":"/Volumes/MCastile/Castile/MForward/Garbage to Better Music/DataClean/Bounces/Wagner_RH/RH_4.wav"}
{"path":"/Volumes/MCastile/Castile/MForward/Garbage to Better Music/DataClean/Bounces/Wagner_RH/RH_5.wav"}
{"path":"/Volumes/MCastile/Castile/MForward/Garbage to Better Music/DataClean/Bounces/Wagner_RH/RH_6.wav"}
{"path":"/Volumes/MCastile/Castile/MForward/Garbage to Better Music/Random Gen/Bounces/Freely Player R.H. Audio/Freely_RH_33.wav"}
{"path":"/Volumes/MCastile/Castile/MForward/Garbage to Better Music/Random Gen/Bounces/Freely Player R.H. Audio/Freely_RH_50.wav"}
```

Output files:

| Axes name | Full name |
| --- | --- |
| CE | Content Enjoyment |
| CU | Content Usefulness |
| PC | Production Complexity |
| PQ | Production Quality |

```jsx
{"CE": 6.90223503112793, "CU": 7.987825870513916, "PC": 2.6831536293029785, "PQ": 6.920309543609619}
{"CE": 7.216231346130371, "CU": 8.074430465698242, "PC": 2.7877769470214844, "PQ": 7.280551433563232}
{"CE": 6.9720611572265625, "CU": 8.02517032623291, "PC": 2.8557169437408447, "PQ": 7.403813362121582}
{"CE": 7.083011627197266, "CU": 8.060602188110352, "PC": 2.6115031242370605, "PQ": 7.59159517288208}
{"CE": 7.24672794342041, "CU": 8.028478622436523, "PC": 3.18192458152771, "PQ": 7.301663875579834}
{"CE": 6.325119972229004, "CU": 7.707026481628418, "PC": 2.347975730895996, "PQ": 7.000254154205322}
{"CE": 7.153213977813721, "CU": 8.055963516235352, "PC": 2.8343513011932373, "PQ": 7.124020576477051}
{"CE": 7.262824058532715, "CU": 7.852158546447754, "PC": 3.190176010131836, "PQ": 7.325965404510498}
{"CE": 7.251339912414551, "CU": 7.987005233764648, "PC": 2.405369997024536, "PQ": 7.766765594482422}
```

Notes: if what I tried is correct, I will form them to an excel file and analysis these results. 

### Tried MusicGen Colab Version to generate music with different prompt style
https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/MusicGen.ipynb#scrollTo=5495f568-51ca-439d-b47b-8b52e89b78f1

### Tried MusicGen locally, environment settled

- API folder, which prompt sample - https://drive.google.com/drive/folders/1lbNEaFlvKMymaVumcK2SC7uk91sHknHW?usp=share_link
- successfully ran MusicGen-small, with the output 0.wav, which is a text-to-music gen model - code in folder
    
    ```python
    import torch
    import torchaudio
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write
    from torch.utils import _pytree # might need to call or not depends on env
    
    model = MusicGen.get_pretrained('small')
    model.set_generation_params(duration=8)  # generate 8 seconds.
    
    descriptions = ['happy piano']
    
    wav = model.generate(descriptions)  # generates 3 samples.
    
    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    ```
    
- unable to ran MusicGen-melody, because computer memory is too small, asking Chun for help - code in folder
    
    ```python
    import torch
    import torchaudio
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write
    from torch.utils import _pytree
    
    model = MusicGen.get_pretrained('melody')
    model.set_generation_params(duration=8)  # generate 8 seconds.
    wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples, might change to other numbers
    descriptions = ['happy piano']
    wav = model.generate(descriptions)  # generates
    
    melody, sr = torchaudio.load('/path/to/bolero_ravel.mp3')
    # generates using the melody from the given audio and the provided descriptions.
    wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)
    
    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    ```
    

env code:

```python
python -m pip install 'torch==2.1.0'
python -m pip install setuptools wheel
python -m pip install -U audiocraft  # stable release
conda install "ffmpeg<5" -c conda-forge
pip install "numpy==1.26.4" --force-reinstall # version collision 
```

### Notes for Ziyu:

- Tried MusicGen locally, environment settled - after asking DeepSeek in many different ways, but still not find where to generate loss while generating music in the inference stage.  I know how to generate loss while using testing/prompting data in evaluation stage, should I build a loss function? or should I using the loss or metrics API below? or should I find other verifiers that built upon LLM to give any piece of music a score? Or should I collect all my audio data as the test set to run at the test time to get the loss from eval?
    - Loss related classes and functions. In particular the loss balancer from EnCodec, and the usual spectral losses. - https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/losses/index.html
    - Or evaluated by metrics like CLAP score, FAD, KLD, Visqol, Chroma similarity, etc. - https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/metrics/index.html
    - Attached some results from DeepSeek, which I tried but failed so far:
        
        ```python
        from magenta.models.music_transformer import MusicTransformerModel
        
        # 加载预训练模型
        model = MusicTransformerModel()
        model.initialize()
        
        # 输入MIDI（转换为NoteSequence）
        midi_path = "input.mid"
        note_sequence = midi_io.midi_file_to_note_sequence(midi_path)
        
        # 计算 log-likelihood
        log_likelihood = model.evaluate(note_sequence)["log_likelihood"]
        print(f"模型对输入MIDI的似然值: {log_likelihood}")
        ```
        
        ```python
        from audiocraft.models import MusicGen
        
        model = MusicGen.get_pretrained("small")
        audio = load_audio("input.wav")  # 形状 [1, T]
        
        # 编码音频并计算重建损失
        encoded = model.compression_model.encode(audio)
        loss = model.loss(encoded)  # 例如均方误差
        print(f"重建损失: {loss.item()}")
        ```
### Notes for Chun:

- Weights will be downloaded while first time running the API
- Test sample bolero_ravel.mp3 is 32kHz Stereo, which is fine, if successes on your end, I will convert other samples to the same.
- Gen sample 0.wav is 32kHz Mono. Both could be use as prompt for MusicGen-melody
- Folder attached again, in case you missed - https://drive.google.com/drive/u/1/folders/1lbNEaFlvKMymaVumcK2SC7uk91sHknHW

# Testing - 4/10/2025 - Log

### **Summary：**

After clarifying the details with Ziyu and Chun, I calculated the loss for the audio input *0.wav*  using the MusicGen model. I’ve also shared some useful links and a Git repository with Chun for testing the musician-melody model. In the upcoming week, my tasks will include preparing audio data to align with different music generative models, writing Python code to compute the loss for audio examples in batch, and organizing the loss calculations into an Excel sheet for further analysis.

**Previous Log：**

[Testing - 4/9/2025 - Log](https://www.notion.so/Testing-4-9-2025-Log-1d1fc45209738069814dfe32bdea3097?pvs=21) 

**Useful Links：**

AudioCrafe-MusicGen： https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md

Audio-Diffusion-Pytorch： https://github.com/archinetai/audio-diffusion-pytorch

MusicGen Melody Hugging Face：https://huggingface.co/docs/transformers/main/en/model_doc/musicgen_melody#audio-conditional-generation

MusicGen - Melody - 1.5B： https://huggingface.co/facebook/musicgen-melody

MusicGen - Stereo - Small - 300M：https://huggingface.co/facebook/musicgen-stereo-small

**Git Repo：**

Chun, feel free to clone the git, and try with the *musicgen-melody_demo* (more guidance in yesterday’s log) and the *musicgen-melody-loss*. Have fun! 

[https://github.com/YoEv/LossCal](https://github.com/YoEv/LossCal)

### **MusicGen-small Loss Calculation：**

This program first tokenizes the input audio, then processes it through the model to generate outputs, and finally computes the negative log-likelihood (NLL) loss between the generated outputs and the target data.

```python
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read
from audiocraft.modules.conditioners import ConditioningAttributes

model = MusicGen.get_pretrained('facebook/musicgen-small')

audio_tensor, sr = audio_read('/path/to/0.wav')
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

```

**Loss Output：**

```python

# input_tokens.shape: torch.Size([1, 4, 399])
# target_tokens.shape: torch.Size([1, 4, 399])
# logits.shape: torch.Size([1596, 2048])
# Negative Log-Likelihood Loss: 5.701778411865234
```

