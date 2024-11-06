import argparse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
from glob import glob
import torch
from tqdm import tqdm
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

def WavLM_embeddings(wav_directory, save_directory):
    ensure_directory_exists(save_directory)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    speaker_encoder_wavLM = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv').to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-sv")
    
    for wav_file in os.listdir(wav_directory):
        if wav_file.endswith('.wav'):
            embedding_filename = os.path.splitext(wav_file)[0] + '.pt'
            save_path = os.path.join(save_directory, embedding_filename)
            
            if os.path.exists(save_path):
                print(f"{embedding_filename} already exists. Skipping...")
                continue

            try:
                print(f"Processing {wav_file}")
                wav_path = os.path.join(wav_directory, wav_file)
                audio, sr = librosa.load(wav_path, sr=16000)

                if len(audio) < 16000:
                    pad_length = 16000 - len(audio)
                    audio = np.pad(audio, (0, pad_length), 'constant', constant_values=0)
                    print(f"Padded {wav_file} with {pad_length} samples")

                input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)

                with torch.no_grad():
                    WavLM_embed = speaker_encoder_wavLM(input_values).embeddings.to("cpu")

                torch.save(WavLM_embed, save_path)
                print(f"Saved embedding for {wav_file}")

            except Exception as e:
                print(f"Error processing {wav_file}: {e}")

def Emotion2Vec_embeddings(wav_directory, save_directory):
    # Initialize the pipeline
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model="iic/emotion2vec_plus_base"
    )
    
    # Perform emotion recognition inference for each file using tqdm for progress bar
    for wav_file in tqdm(wav_directory, desc="Processing WAV files"):
        try:
            base_name = os.path.splitext(os.path.basename(wav_file))[0]
            save_path = os.path.join(save_directory, f"{base_name}.pt")

            # Perform inference on the individual file
            tgt_result = inference_pipeline([wav_file], granularity="utterance")

            for i in tgt_result:
                file_name = i["key"]
                tgt_emb = i["feats"]

                # Convert tgt_emb to a tensor, move to cpu and detach
                tgt_emb_tensor = torch.tensor(tgt_emb).cpu().detach()

                # Save the tensor to a .pt file
                torch.save(tgt_emb_tensor, save_path)
        
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_directory", type=str, default="/workspace/choddeok/hd0/dataset/ESD")
    parser.add_argument("--wavlm_save_directory", type=str, default="/workspace/choddeok/hd0/dataset/ESD_emb/WavLM")
    parser.add_argument("--emotion2vec_save_directory", type=str, default="/workspace/choddeok/hd0/dataset/ESD_emb/Emotion2Vec")
    args = parser.parse_args()
    
    WavLM_embeddings(args.wav_directory, args.save_directory)
    Emotion2Vec_embeddings(args.wav_directory, args.save_directory)