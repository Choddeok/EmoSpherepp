# EmoSphere++: Emotion-Controllable Zero-Shot Text-to-Speech via Emotion-Adaptive Spherical Vector <br><sub>The official implementation of EmoSphere++</sub>

<div align="center">
  <img src="https://github.com/user-attachments/assets/87d79c36-790e-488d-8837-9ef40318dc57" width="80%" />

</div>

## [Paper 📄](https://ieeexplore.ieee.org/document/10965917)|[Demo 🎧](https://choddeok.github.io/EmoSphere-Demo/)

**The official Pytorch implementation of EmoSphere++ (IEEE Transactions on Affective Computing 2025)**

**Deok-Hyeon Cho, Hyung-Seok Oh, Seung-Bin Kim, Seong-Whan Lee**


## Abstract
Emotional text-to-speech (TTS) technology has achieved significant progress in recent years; however, challenges remain owing to the inherent complexity of emotions and limitations of the available emotional speech datasets and models. Previous studies typically relied on limited emotional speech datasets or required extensive manual annotations, restricting their ability to generalize across different speakers and emotional styles. In this paper, we present EmoSphere++, an emotion-controllable zero-shot TTS model that can control emotional style and intensity to resemble natural human speech. We introduce a novel emotion-adaptive spherical vector that models emotional style and intensity without human annotation. Moreover, we propose a multi-level style encoder that can ensure effective generalization for both seen and unseen speakers. We also introduce additional loss functions to enhance the emotion transfer performance for zero-shot scenarios. We employ a conditional flow matching-based decoder to achieve high-quality and expressive emotional TTS in a few sampling steps. Experimental results demonstrate the effectiveness of the proposed framework.

## Training Procedure

### Library
- <a href="https://www.python.org/">Python</a> >= 3.10
- <a href="https://pytorch.org/get-started/pytorch-2.0/">PyTorch</a> >= 2.0 (Recommand)
- <a href="https://pytorch.org/get-started/pytorch-2.0/">CUDA</a> >= 11.6


```bash
  # Docker image
  DOCKER_IMAGE=nvcr.io/nvidia/pytorch:24.02-py3
  docker pull $DOCKER_IMAGE

  # Set docker config
  CONTAINER_NAME=YOUR_CONTAINER_NAME
  SRC_CODE=YOUR_CODE_PATH
  TGT_CODE=DOCKER_CODE_PATH
  SRC_DATA=YOUR_DATA_PATH
  TGT_DATA=DOCKER_DATA_PATH
  SRC_CKPT=YOUR_CHECKPOINT_PATH
  TGT_CKPT=DOCKER_CHECKPOINT_PATH
  SRC_PORT=6006
  TGT_PORT=6006
  docker run -itd --ipc host --name $CONTAINER_NAME -v $SRC_CODE:$TGT_CODE -v $SRC_DATA:$TGT_DATA -v $SRC_CKPT:$TGT_CKPT -p $SRC_PORT:$TGT_PORT --gpus all --restart=always $DOCKER_IMAGE
  docker exec -it $CONTAINER_NAME bash

  apt-get update
  # Install tmux
  apt-get install tmux -y
  # Install espeak
  apt-get install espeak -y

  # Clone repository in docker code path
  git clone https://github.com/Choddeok/EmoSpherepp.git

  pip install -r requirements.txt
```

### Vocoder
The BigVGAN 16k checkpoint will be released at a later date. In the meantime, please train using the official BigVGAN implementation or use the official HiFi-GAN checkpoint.
- [[HiFi-GAN]](https://github.com/jik876/hifi-gan)
- [[BigVGAN]](https://github.com/NVIDIA/BigVGAN)

------
### 1. Preprocess data
- Modify the config file to fit your environment.
- We use ESD database, which is an emotional speech database that can be downloaded here: https://hltsingapore.github.io/ESD/.

#### a) VAD Analysis
- Steps for emotion-specific centroid extraction with VAD analysis
```bash
sh Analysis.sh
```

#### b) Preprocessing
- Steps for embedding extraction and binary dataset creation (https://huggingface.co/microsoft/wavlm-base-sv, https://huggingface.co/emotion2vec/emotion2vec_plus_large)

Before running the script below, please set the following paths in preprocessing.sh:

- wav_directory: path to your original wav files

- wavlm_save_directory: path to save extracted WavLM embeddings

- emotion2vec_save_directory: path to save Emotion2Vec embeddings

For binary dataset creation, we follow the pipeline from [[NATSpeech]](https://github.com/NATSpeech/NATSpeech).

**Note: You do not need to run MFA (Montreal Forced Aligner) for our setting.**

```bash
sh preprocessing.sh
```

------
### 2. Training TTS module and Inference  
```bash
sh train_run.sh
```

------
### 3. Pretrained checkpoints
- TTS module trained on 11M [[Download]](https://works.do/xO6ZtDB)

## Citation
```bash
@ARTICLE{10965917,
  author={Cho, Deok-Hyeon and Oh, Hyung-Seok and Kim, Seung-Bin and Lee, Seong-Whan},
  journal={IEEE Transactions on Affective Computing}, 
  title={EmoSphere++: Emotion-Controllable Zero-Shot Text-to-Speech via Emotion-Adaptive Spherical Vector}, 
  year={2025},
  volume={},
  number={},
  pages={1-16},
  keywords={Vectors;Text to speech;Aerospace electronics;Psychology;Complexity theory;Interpolation;Emotion recognition;Annotations;Wheels;Training;Emotional speech synthesis;emotion transfer;emotion style and intensity control;zero-shot text-to-speech},
  doi={10.1109/TAFFC.2025.3561267}}
```
## Acknowledgements
**Our codes are based on the following repos:**
* [NATSpeech](https://github.com/NATSpeech/NATSpeech)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [BigVGAN](https://github.com/NVIDIA/BigVGAN)
