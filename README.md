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
        
    
    - Attached are some other results from DeepSeek I haven’t try yet:
        
        Prompt: 
        
        好，我们重新思考一下这个问题。
        第一个问题：
        我给一个模型我的audio 或者 midi input，然后让那个音乐生成模型生成一个generated audio，然后呢我这个audio或者midi input都是从原来的一首音乐的original audio产生的，那么我怎么在这个音乐生成模型的内部去对比这个generated audio和original audio的loss呢？或者一个返回的likelihood？
        第二个问题：
        生成模型在生成的过程中，会给一个loss或者likelihood吗？按照我的前辈的说法是，是每一次生成都会返回一个loss。
        第三个问题：
        如果不是生成模型，而是别人已经做好的打分器的话，会返回这样的loss吗？
        
        Results: 
        
        你的三个问题都非常核心，涉及生成模型的内部机制和评估方法。我逐一分析：
        
        ### 第一个问题：模型内部如何对比生成和原作的loss/likelihood？
        
        这取决于模型的类型和设计意图：
        
        1. **自回归模型**（如Music Transformer）：
            - 训练时：通过交叉熵损失预测下一个音符/音频片段
            - **生成时**：若想对比原曲，需要强制让模型在生成过程中每一步都访问原作的ground truth数据，计算每个生成步骤的likelihood（类似teacher forcing）
            - 实现方法：修改生成逻辑，在采样时同时计算每个token的log概率并累积
        2. **VAE/GAN类模型**：
            - 传统VAE训练时计算重构损失（原曲与重建音频的差异），但生成时无法直接对比原曲
            - 若需生成时对比，需改造模型架构，例如：
                - 在潜在空间加入原曲的编码特征作为condition
                - 设计双通道结构：一条生成路径，一条重构路径共享encoder
        3. **Diffusion模型**：
            - 训练时通过预测噪声的MSE损失隐式学习数据分布
            - 生成时可通过计算反向扩散过程中每个step的预测噪声与原曲加噪后的噪声对比（需要访问原曲的加噪版本）
        
        **关键难点**：标准生成过程不需要访问原曲，若强行引入原曲对比，可能造成信息泄漏（相当于用原曲当测试集的ground truth）。更合理的做法是在训练时设计特殊的loss约束。
        
        ### 第二个问题：生成过程是否返回loss？
        
        常规情况：
        
        - **训练时**：必然有loss计算（反向传播用）
        - **生成/推理时**：标准流程**不计算loss**，因为：
            - loss需要ground truth，而生成时没有正确答案
            - 生成是开放式的采样过程
        
        例外情况：
        
        1. **概率输出模型**（如自回归模型）：
            - 可记录每个生成步骤的token概率 → 计算序列的**对数似然**(log-likelihood)
            - 例如：GPT类模型生成时保留每个位置的logits，事后计算perplexity
        2. **强化学习框架**：
            - 若使用类似PPO的算法，生成时会计算reward（可视为一种likelihood proxy）
        3. **Teacher Forcing调试模式**：
            - 在开发阶段，可强制让生成过程使用原曲作为输入，此时能计算loss（但本质上已不是真正的生成）
        
        ### 第三个问题：外部打分器如何工作？
        
        已有评估工具可分为两类：
        
        **1. 基于特征的评估器**（无需训练）：
        
        - **音频相似度指标**：
            - FAD（Frechet Audio Distance）：对比生成/原曲在VGGish特征空间的分布
            - Chroma-CENS、MFCC距离等
        - **音乐特性指标**：
            - 音高分布KL散度
            - 节奏相似度
            - 和弦进行匹配度
        
        **2. 预训练判别模型**：
        
        - **分类器打分**：
            - 训练一个"音乐质量分类器"，用其置信度作为likelihood proxy
            - 例如：用NSynth预训练模型提取特征后计算相似度
        - **对抗性评估**：
            - 使用GAN的判别器输出作为"真实性分数"
        - **对比学习模型**：
            - 如CLAP模型，计算音频-乐谱对齐分数
        
        **典型工具包**：
        
        - `museval`：音乐结构相似度评估
        - `mir_eval`：低级特征对比
        - `audiocraft`（Meta）：内置FAD计算工具
        - `jukemir`：预训练音乐特征提取

### Notes for Chun:

- Weights will be downloaded while first time running the API
- Test sample bolero_ravel.mp3 is 32kHz Stereo, which is fine, if successes on your end, I will convert other samples to the same.
- Gen sample 0.wav is 32kHz Mono. Both could be use as prompt for MusicGen-melody
- Folder attached again, in case you missed - https://drive.google.com/drive/u/1/folders/1lbNEaFlvKMymaVumcK2SC7uk91sHknHW
