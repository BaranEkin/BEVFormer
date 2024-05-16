import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.nn.functional as F
import os

class BEVCapGen(nn.Module):
    def __init__(
        self,
        bev_encoder,
        text_decoder,
        tokenizer,
        bev_feature_size,
        bev_area,
        hidden_size,
        encoder_width,
        prompt="",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        bev_only=False
    ):
        super().__init__()
        self.device = device
        self.bev_only = bev_only

        self.bev_size = bev_feature_size * bev_area

        # Freezing the BEV encoder
        self.bev_encoder = bev_encoder.to(self.device)
        for param in self.bev_encoder.parameters():
            param.requires_grad = False

        if not self.bev_only:
            # FC layer to map BEV feature size to cross attention width
            self.bev_feature_mapper = nn.Linear(bev_feature_size, encoder_width).to(self.device)

            # Projectors for bev and text to a common dimension for contrastive loss
            projection_dim = 512
            self.bev_projector = nn.Linear(self.bev_size, projection_dim).to(self.device)
            self.text_projector = nn.Linear(hidden_size, projection_dim).to(self.device)
            self.logit_scale = nn.Parameter(torch.tensor(2.6592)).to(self.device)

            # Text decoder and tokenizer
            self.text_decoder = text_decoder.to(self.device)
            self.tokenizer = tokenizer

            # BEV-Text Matching Head
            self.btm_head = nn.Linear(hidden_size, 2).to(self.device)
            
            # Prompt
            self.prompt = prompt
            self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, data, caption):

        new_data = {}
        new_data["img_metas"] = data["img_metas"][0].data
        new_data["img"] = [data["img"][0].data[0].to(self.device)]


        bev_embeds = self.bev_encoder(return_loss=False, rescale=True, only_bev_embed=True, **new_data)
        bev_features_for_ca = self.bev_feature_mapper(bev_embeds)
        
        bev_atts = torch.ones(bev_features_for_ca.size()[:-1], dtype=torch.long).to(self.device)

        text = self.tokenizer(caption,
            padding="longest",
            truncation=True,
            max_length=40,
            return_tensors="pt",
        ).to(self.device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

        decoder_output, text_embeds = self.text_decoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=bev_features_for_ca,
            encoder_attention_mask=bev_atts,
            labels=decoder_targets,
            return_dict=True,
        )

        # IMAGE(BEV) - TEXT CONTRASTIVE LOSS --------------------------
        
        batch_size = bev_embeds.shape[0]
        bev_embeds = bev_embeds.reshape(batch_size, self.bev_size)
        bev_embeds_projected = self.bev_projector(bev_embeds)

        text_mean_embeds = torch.mean(text_embeds, dim=1, keepdim=True).squeeze()
        text_embeds_projected = self.text_projector(text_mean_embeds)
        
        
        text_embeds_projected = text_embeds_projected / text_embeds_projected.norm(dim=1, keepdim=True)
        bev_embeds_projected = bev_embeds_projected / bev_embeds_projected.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_bev = (bev_embeds_projected @ text_embeds_projected.t()) * logit_scale
        logits_per_text = logits_per_bev.t()

        cl_targets = torch.arange(batch_size, dtype=torch.long).to(self.device)
        contrastive_loss = (F.cross_entropy(logits_per_bev, cl_targets) 
                            + F.cross_entropy(logits_per_text, cl_targets)) / 2
        
        # -----------------------------------------------------------------
        

        return {"contrastive_loss": contrastive_loss,
                "lm_loss": decoder_output.loss}
    
    def get_bev_embeds(self, data):
        new_data = {}
        new_data["img_metas"] = data["img_metas"].data[0] # img_metas = [{0: {}, 1:{}, 2:{}}]
        new_data["img"] = data["img"].data[0].to(self.device) # img = tensor (bs=1, qs=3, mview=6, c=3, h=480, w=800)

        bev_embeds, det_centers = self.bev_encoder(only_bev=True, **new_data)

        det_centers_bev = torch.trunc((det_centers + 51.2) / 2.048).int()
        det_idx = det_centers_bev[:, 0] + (det_centers_bev[:, 1] * 50)

        det_embeds = torch.index_select(bev_embeds.squeeze(), 0, det_idx.flatten().to(self.device))
        
        """
        # Visualization
        A = torch.zeros((50, 50))
        A[det_centers_bev[:, 1].long(), det_centers_bev[:, 0].long()] += 1
        """

        return bev_embeds, det_embeds

    def generate(
        self,
        data,
        max_length=30,
        min_length=10,
        top_p=0.9,
    ):
        new_data = {}
        new_data["img_metas"] = data["img_metas"][0].data
        new_data["img"] = [data["img"][0].data[0].to(self.device)]
        
        bev_embeds = self.bev_encoder(return_loss=False, rescale=True, only_bev_embed=True, **new_data)
        bev_embeds = self.bev_feature_mapper(bev_embeds)
        

        bev_atts = torch.ones(bev_embeds.size()[:-1], dtype=torch.long).to(self.device)
        model_kwargs = {
            "encoder_hidden_states": bev_embeds,
            "encoder_attention_mask": bev_atts,
        }

        prompt = [self.prompt] * 1 # batch size
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        # nucleus sampling
        outputs = self.text_decoder.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.1,
            **model_kwargs
        )

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt) :])
        return captions

    def load_blip_decoder_ckpt(self, ckpt_path):

        ckpt = torch.load(ckpt_path)
        ckpt_dict = ckpt["model"]

        self_dict = self.state_dict()
        ckpt_dict = {k: v for k, v in ckpt_dict.items() if k in self_dict}
        self_dict.update(ckpt_dict)
        self.load_state_dict(self_dict)




