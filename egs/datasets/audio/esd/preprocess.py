from data_gen.tts.base_preprocess import BasePreprocessor


class esdPreprocess(BasePreprocessor):
    def meta_data(self):
        for l in open(f'{self.raw_data_dir}/esd_text_emo.txt').readlines():
            wav_fn, txt, _ = l.strip().split("|")
            item_name_base = wav_fn.strip().split("/")[-1]
            item_name = item_name_base.strip().split(".")[0]
            spk_name = item_name.strip().split("_")[0]
            emo = wav_fn.strip().split("/")[-2]
            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name, 'emo_name': emo}
