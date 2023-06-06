import torch
import torch.nn as nn
from .modeling_t5 import T5ForConditionalGeneration
from .vit import VisionTransformer
from transformers import T5Tokenizer
from transformers.modeling_outputs import (
    BaseModelOutput,
)

def _get_tokenizer(tokenizer_path, num_bins=0):
    if 't5' in tokenizer_path:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if num_bins:
            new_tokens = ["<time=" + str(i) + ">" for i in range(num_bins)]
            tokenizer.add_tokens(list(new_tokens))
    else:
        raise NotImplementedError(tokenizer_path)
    return tokenizer

class Vid2Seq(torch.nn.Module):
    def __init__(self,
                 t5_path,
                 num_features=100,
                 embed_dim=768,
                 depth=12,
                 heads=12,
                 mlp_dim=2048,
                 vis_drop=0.,
                 tokenizer=None,
                 enc_drop=0.,
                 dec_drop=0.1,
                 use_speech=True,
                 use_video=True,
                 num_bins=100,
                 label_smoothing=0.1):
        super().__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(encoder_dropout=enc_drop, decoder_dropout=dec_drop, label_smoothing=label_smoothing,
                                                                   pretrained_model_name_or_path=t5_path, local_files_only=True, is_gated_act="v1_1" in t5_path)
        self.t5_model.resize_token_embeddings(len(tokenizer) - num_bins)  # remove the weights of the 28 tokens that are not used (32128 vs 32100 in the tokenizer)
        self.t5_model.resize_token_embeddings(len(tokenizer))  # add time tokens
        self.visual_encoder = VisionTransformer(num_features=num_features,
                                                embed_dim=embed_dim,
                                                depth=depth,
                                                num_heads=heads,
                                                mlp_dim=mlp_dim,
                                                qkv_bias=True,
                                                qk_scale=None,
                                                drop_rate=vis_drop,
                                                attn_drop_rate=vis_drop,
                                                norm_layer=nn.LayerNorm)
        self.t5_tokenizer = tokenizer
        self.use_speech = use_speech
        self.use_video = use_video
        self.proj_v2t = None
        if self.t5_model.model_dim != 768:
            self.proj_v2t = nn.Linear(768, self.t5_model.model_dim)

    def forward(self, video, input_tokenized, output_tokenized):
        if self.use_video:
            if isinstance(video, dict):  # cached
                video, atts_vis = video["video"], video["atts_vis"]
            else:
                video = self.visual_encoder(video)  # B T D
                if self.proj_v2t is not None:
                    video = self.proj_v2t(video)
                atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)
            video_dict = {"video": video, "atts_vis": atts_vis}
        else:
            video_dict = None
        if self.use_speech:
            text = self.t5_model.encoder.embed_tokens(input_tokenized['input_ids'])  # B L D
            encoded = self.t5_model.encoder(
                attention_mask=input_tokenized['attention_mask'],
                inputs_embeds=text,
            )

        if self.use_video and self.use_speech:
            encoded.last_hidden_state = torch.cat([video, encoded.last_hidden_state], dim=1)
            encoder_atts = torch.cat([atts_vis, input_tokenized['attention_mask']], dim=1)
        elif self.use_video:
            encoded = BaseModelOutput(last_hidden_state=video)
            encoder_atts = atts_vis
        elif self.use_speech:
            encoder_atts = input_tokenized['attention_mask']

        targets = output_tokenized['input_ids'].masked_fill(
            output_tokenized['input_ids'] == self.t5_tokenizer.pad_token_id, -100
        )
        outputs = self.t5_model(
            encoder_outputs=encoded,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokenized['attention_mask'],
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}, video_dict

    @torch.no_grad()
    def generate(
            self,
            video,
            input_tokenized,
            use_nucleus_sampling=False,
            num_beams=4,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
    ):
        """
        Args:
            video (torch.Tensor): A tensor of shape (batch_size, T, D)
            input_tokenized (torch.Tensor): A tensor of shape (batch_size, L)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if self.use_video:
            video = self.visual_encoder(video)  # B T D
            if self.proj_v2t is not None:
                video = self.proj_v2t(video)
            atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)
        if self.use_speech:
            text = self.t5_model.encoder.embed_tokens(input_tokenized['input_ids'])  # B L D
            encoded = self.t5_model.encoder(
                attention_mask=input_tokenized['attention_mask'],
                inputs_embeds=text,
            )

        if self.use_video and self.use_speech:
            encoded.last_hidden_state = torch.cat([video, encoded.last_hidden_state], dim=1)
            encoder_atts = torch.cat([atts_vis, input_tokenized['attention_mask']], dim=1)
        elif self.use_video:
            encoded = BaseModelOutput(last_hidden_state=video)
            encoder_atts = atts_vis
        elif self.use_speech:
            encoder_atts = input_tokenized['attention_mask']

        outputs = self.t5_model.generate(
                encoder_outputs=encoded,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
        )
        output_text = self.t5_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return output_text
