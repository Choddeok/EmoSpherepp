from torch import nn
from models.commons.layers import Embedding
from models.commons.nar_tts_modules import DurationPredictor, LengthRegulator
from models.commons.transformer import FastSpeechEncoder, FastSpeechDecoder
from models.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states


class FastSpeech(nn.Module):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = hparams
        self.hidden_size = hparams["hidden_size"]
        self.encoder = FastSpeechEncoder(
            dict_size,
            hparams["hidden_size"],
            hparams["encoder_layers"],
            hparams["encoder_ffn_kernel_size"],
            num_heads=hparams["num_heads"],
        )
        self.decoder = FastSpeechDecoder(
            hparams["hidden_size"],
            hparams["decoder_layers"],
            hparams["decoder_ffn_kernel_size"],
            hparams["num_heads"],
        )
        self.out_dims = hparams["audio_num_mel_bins"] if out_dims is None else out_dims
        self.mel_out = nn.Linear(self.hidden_size, self.out_dims, bias=True)

        if hparams["use_spk_id"]:
            self.spk_id_proj = Embedding(hparams["num_spk"], self.hidden_size)
        if hparams["use_spk_embed"]:
            self.spk_embed_proj = nn.Linear(256, self.hidden_size, bias=True)

        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=self.hidden_size,
            n_layers=hparams["dur_predictor_layers"],
            dropout_rate=hparams["predictor_dropout"],
            kernel_size=hparams["dur_predictor_kernel"],
        )
        self.length_regulator = LengthRegulator()

    def forward(
        self,
        txt_tokens,
        mel2ph=None,
        spk_embed=None,
        spk_id=None,
        infer=False,
        **kwargs
    ):
        ret = {}
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        style_embed = self.forward_style_embed(spk_embed, spk_id)

        # add dur
        dur_inp = (encoder_out + style_embed) * src_nonpadding
        mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret, infer)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(encoder_out, mel2ph)

        # decoder input
        ret["decoder_inp"] = decoder_inp = (decoder_inp + style_embed) * tgt_nonpadding

        ret["mel_out"] = self.forward_decoder(
            decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs
        )
        return ret

    def forward_style_embed(self, spk_embed=None, spk_id=None):
        # add spk embed
        style_embed = 0
        if self.hparams["use_spk_embed"]:
            style_embed = style_embed + self.spk_embed_proj(spk_embed)[:, None, :]
        if self.hparams["use_spk_id"]:
            style_embed = style_embed + self.spk_id_proj(spk_id)[:, None, :]
        return style_embed

    def forward_dur(self, dur_input, mel2ph, txt_tokens, ret, infer=False):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = txt_tokens == 0
        dur_input = dur_input.detach()
        dur = self.dur_predictor(dur_input, src_padding)
        ret["dur"] = dur
        if infer:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret["mel2ph"] = mel2ph = clip_mel2token_to_multiple(
            mel2ph, self.hparams["frames_multiple"]
        )
        return mel2ph

    def forward_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x)
        x = self.mel_out(x)
        return x * tgt_nonpadding
