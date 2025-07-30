# clip-vit-base-patch16 is a pretrained CLIP model with released weights: https://huggingface.co/openai/clip-vit-base-patch16 with license http://www.apache.org/licenses/LICENSE-2.0.

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import utils
from utils import *
from typing import Any, Optional, Tuple, Union
import pandas as pd
import torch.nn.functional as F
from .clipsym_modeling.modeling import *
from transformers.models.clip.modeling_clip import CLIPPreTrainedModel, CLIPConfig, CLIPTextTransformer, CLIPVisionTransformer

pretrained_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

class CLIPSym(nn.Module):
    def __init__(self, args, prompts, reduce_dim=64):
        super(CLIPSym, self).__init__()
        self.args = args
        self.prompts = prompts
        
        # not pretrain ...
        # pretrained
        self.config = pretrained_clip.config
        # self.text_config = self.config.text_config
        # self.vision_config = self.config.vision_config
        
        self.clip = CLIPModelV2(self.config)
        
        if not self.args.clip_scratch:
            # use pretrained clip model to initialize every module in self.clip
            self.clip.load_state_dict(pretrained_clip.state_dict())
        
        # fixed cond embedding for frozen text encoder
        self.fixed_cond_embed = None
        encoding = processor.tokenizer(prompts, padding='max_length', return_tensors='pt')
        self.input_ids = encoding['input_ids']
        self.attention_mask = encoding['attention_mask']
        with torch.no_grad():
            self.fixed_cond_embed = self.clip.get_text_features(self.input_ids, attention_mask=self.attention_mask)
        
        self.extract_layers = [3, 6, 9]

        self.reduce_dim = reduce_dim
        self.decoder = CLIPSymDecoder(self.config, num_prompts=args.num_prompts, equivariant_upsampler=args.equivariant_upsampler, extract_layers=self.extract_layers, reduce_dim=reduce_dim)

        self.sigmoid = nn.Sigmoid()


    def get_conditional_embeddings(
        self,
        batch_size: int = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if input_ids is not None:
            if not self.args.freeze_text_encoder:
                with torch.no_grad():
                    conditional_embeddings = self.clip.get_text_features(
                        input_ids, attention_mask=attention_mask
                    )
            else:
                conditional_embeddings = self.clip.get_text_features(
                    input_ids, attention_mask=attention_mask
                )

        return conditional_embeddings

    
    def forward(self, img, axis, lbl, a_lbl=None, sym_type='reflection', vis_only=False, return_dict=True
                ):
                
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # True
        pixel_values = img
        self.input_ids = self.input_ids.to(img.device)
        self.attention_mask = self.attention_mask.to(img.device)
        self.fixed_cond_embed = self.fixed_cond_embed.to(img.device)

        # step 1: forward the query images through the frozen CLIP vision encoder
        if self.args.freeze_img_encoder:
            with torch.no_grad():
                # print(">>> no grad img encoder")
                vision_outputs = self.clip.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=False,
                    output_hidden_states=True,  # we need the intermediate hidden states
                    return_dict=return_dict,
                ) # (pixel_values needs to have the same height and width)
                pooled_output = self.clip.visual_projection(vision_outputs[1]) # (bs, 512)
        else:
            # print(">>> grad img encoder")
            vision_outputs = self.clip.vision_model(
                pixel_values=pixel_values,
                output_attentions=False,
                output_hidden_states=True,  # we need the intermediate hidden states
                return_dict=return_dict,
            ) # (bs, 768)
            pooled_output = self.clip.visual_projection(vision_outputs[1]) # (bs, 512)
        
        hidden_states = vision_outputs.hidden_states if return_dict else vision_outputs[2] # len 13, extract_layers = 3, 6, 9 

        # we add +1 here as the hidden states also include the initial embeddings
        activations = [hidden_states[i + 1] for i in self.extract_layers]

         # step 2: compute conditional embeddings, either from text, images or an own provided embedding
        if not self.args.freeze_text_encoder:
            conditional_embeddings = self.get_conditional_embeddings(
                batch_size=pixel_values.shape[0], # is not important
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
            ) # (n_prompts, 512)
        else:
            conditional_embeddings = self.fixed_cond_embed # (n_prompts, 512)
        
        decoder_outputs = self.decoder(activations, conditional_embeddings)

        axis_out = decoder_outputs.logits if return_dict else decoder_outputs[0] # (bs * len(prompts), 352, 352)

        axis_out = F.interpolate(axis_out, size=lbl.size()[2:], mode='bilinear', align_corners=True)

        gt = lbl
        axis_loss = utils.sigmoid_focal_loss(axis_out, gt, alpha=self.args.fl_alpha)

        axis_out = self.sigmoid(axis_out)
        
        loss = axis_loss.mean()

        return axis_out, loss, (axis_loss, axis_loss),





        
        