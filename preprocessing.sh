CUDA_VISIBLE_DEVICES=0 python embedding_extract.py --wav_directory "/workspace/choddeok/hd0/dataset/ESD" --wavlm_save_directory "/workspace/choddeok/hd0/dataset/ESD_emb/WavLM" --emotion2vec_save_directory "/workspace/choddeok/hd0/dataset/ESD_emb/WavLM"


CUDA_VISIBLE_DEVICES=0 python align_and_binarize.py --config egs/datasets/audio/esd/fs2_orig.yaml