from torch.nn import functional as F
from tasks.tts.speech_base import SpeechBaseTask
from utils.commons.tensor_utils import tensors_to_scalars
import torch
import numpy as np

class EmoSpherepp(SpeechBaseTask):
    def __init__(self):
        super(EmoSpherepp, self).__init__()

    def forward(self, sample, infer=False, *args, **kwargs):
        hparams = self.hparams
        x = sample["txt_tokens"]  # [B, T_t]
        x_lengths = sample["txt_lengths"]
        y = sample["mels"]  # [B, T_s, 80]
        y_lengths = sample["mel_lengths"]  # [B, T_s, 80]
        
        # spk
        spk_embed = sample.get("WavLM_emb")

        # emo
        emo_id = sample.get("emo_ids")
        inten_vector = sample.get("inten_vectors")
        style_vector = sample.get("style_vectors")
        emotion2vec = sample.get("emotion2vec")
        
        if hparams['intensity'] == "weak":
            new_tensor = torch.tensor([0.1])
            if emo_id.item() == 2:
                new_tensor = torch.tensor([0])
            inten_vector[0] = new_tensor
        elif hparams['intensity'] == "medium":
            new_tensor = torch.tensor([0.5])
            if emo_id.item() == 2:
                new_tensor = torch.tensor([0])
            inten_vector[0] = new_tensor
        elif hparams['intensity'] == "strong":
            new_tensor = torch.tensor([0.9])
            if emo_id.item() == 2:
                new_tensor = torch.tensor([0])
            inten_vector[0] = new_tensor
        
        if hparams['style'] == "I":
            new_tensor = torch.tensor([np.pi/4, np.pi/4])
            if emo_id.item() == 2:
                new_tensor = torch.tensor([0, 0])
            style_vector[0] = new_tensor
        elif hparams['style'] == "III":
            new_tensor = torch.tensor([np.pi/4, -3*np.pi/4])
            if emo_id.item() == 2:
                new_tensor = torch.tensor([0, 0])
            style_vector[0] = new_tensor
        elif hparams['style'] == "V":
            new_tensor = torch.tensor([3*np.pi/4, np.pi/4])
            if emo_id.item() == 2:
                new_tensor = torch.tensor([0, 0])
            style_vector[0] = new_tensor
        elif hparams['style'] == "VII":
            new_tensor = torch.tensor([3*np.pi/4, -3*np.pi/4])
            if emo_id.item() == 2:
                new_tensor = torch.tensor([0, 0])
            style_vector[0] = new_tensor
        elif hparams['style'] == "VIII":
            new_tensor = torch.tensor([3*np.pi/4, -np.pi/4])
            if emo_id.item() == 2:
                new_tensor = torch.tensor([0, 0])
            style_vector[0] = new_tensor
            
        if not infer:
            output = self.model.compute_loss(
                x,
                x_lengths,
                y.transpose(1, 2),
                y_lengths,
                spk=spk_embed,
                emo=emotion2vec,
                inten_vector=inten_vector,
                style_vector=style_vector,
                out_size=self.hparams["out_size"],
            )
            losses = {}
            losses["dur_loss"] = output["dur_loss"]
            losses["prior_loss"] = output["prior_loss"]
            losses["diff_loss"] = output["diff_loss"]
            self.OrthogonalityLoss(output['spks_embed'], output['emos_embed'], losses=losses)

            return losses, output
        else:
            output = self.model(
                x, 
                x_lengths, 
                spk=spk_embed, 
                emo=emotion2vec,
                inten_vector=inten_vector,
                style_vector=style_vector, 
                n_timesteps=100,
                guidance_scale=0)
            return output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs["losses"] = {}
        outputs["losses"], _ = self(sample)
        outputs["nsamples"] = sample["nsamples"]

        if (
            self.global_step % self.hparams["valid_infer_interval"] == 0
            and batch_idx < self.hparams["num_valid_plots"]
        ):
            model_out = self(sample, infer=True)
            self.save_valid_result(sample, batch_idx, model_out)

        outputs = tensors_to_scalars(outputs)
        return outputs

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = self.hparams["audio_sample_rate"]
        gt = sample["mels"]
        pred = model_out["mel_out"]
        prior = model_out["encoder_outputs"]
        attn = model_out["attn"].cpu().numpy()
        
        if batch_idx < 2:
            emo = "Neutral"
        elif batch_idx < 22:
            emo = "Angry"
        elif batch_idx < 42:
            emo = "Happy"
        elif batch_idx < 62:
            emo = "Sad"
        elif batch_idx < 82:
            emo = "Surprise"
        elif batch_idx < 402:
            emo = "Neutral"
        elif batch_idx < 422:
            emo = "Angry"
        elif batch_idx < 442:
            emo = "Happy"
        elif batch_idx < 462:
            emo = "Sad"
        else:
            emo = "Surprise"
        

        self.plot_mel(batch_idx, [gt[0], prior[0], pred[0]], title=f"mel_{batch_idx}_{emo}")
        self.logger.add_image(
            f"plot_attn_{batch_idx}_{emo}", self.plot_alignment(attn[0]), self.global_step
        )

        wav_pred = self.vocoder.spec2wav(pred[0].cpu())
        self.logger.add_audio(f"wav_pred_{batch_idx}_{emo}", wav_pred, self.global_step, sr)

        wav_pred = self.vocoder.spec2wav(prior[0].cpu())
        self.logger.add_audio(f"wav_prior_{batch_idx}_{emo}", wav_pred, self.global_step, sr)

        if self.global_step <= self.hparams["valid_infer_interval"]:
            wav_gt = self.vocoder.spec2wav(gt[0].cpu())
            self.logger.add_audio(f"wav_gt_{batch_idx}_{emo}", wav_gt, self.global_step, sr)
    
    def OrthogonalityLoss(self, speaker_embedding, emotion_embedding, losses=None):
        speaker_embedding_t = speaker_embedding.t()

        dot_product_matrix = torch.matmul(emotion_embedding, speaker_embedding_t)

        emotion_norms = torch.norm(emotion_embedding, dim=1, keepdim=True)
        speaker_norms = torch.norm(speaker_embedding, dim=1, keepdim=True).t()
        normalized_dot_product_matrix = dot_product_matrix / (emotion_norms * speaker_norms)

        ort_loss = torch.norm(normalized_dot_product_matrix, p='fro')**2

        losses['ort_loss'] = 0.02 * ort_loss