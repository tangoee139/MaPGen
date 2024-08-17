import os
import warnings
warnings.filterwarnings("ignore")
import random
from collections import OrderedDict

from transformers import AutoModelForCausalLM
from models.resnet import blip_resnet

import torch
from torch import nn
import torch.nn.functional as F

import pandas as pd
from .group_resampler import PerceiverResampler
from modules.asl_loss import ASLwithClassWeight

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


class Qwen2MRG(nn.Module):
    def __init__(self,                 
                 args,
                 tokenizer=None,
                 ):
        super().__init__()
        self.args = args
        
        vision_width = 2048
        self.visual_encoder = blip_resnet(args) 
        self.resampler = PerceiverResampler(dim=2048, depth=2, num_latents=args.num_latents) 
        self.cls_head = nn.Linear(2048,  self.args.num_label*2)
        nn.init.normal_(self.cls_head.weight, std=0.001)
        if self.cls_head.bias is not None:
            nn.init.constant_(self.cls_head.bias, 0)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.0)



        
        self.tokenizer = tokenizer   
        text_decoder = AutoModelForCausalLM.from_pretrained(
                    args.model_type,
                    torch_dtype=torch.float32,
                )

        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_all_linear_names(text_decoder),
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        self.text_decoder = get_peft_model(text_decoder, lora_config)

        hidden_size = self.text_decoder.config.hidden_size
        self.vision_proj = nn.Sequential(nn.Linear(vision_width, hidden_size),
                                        nn.GELU(),
                                        nn.Linear(hidden_size, hidden_size)
                                        )



        diagnosis_csv = pd.read_csv('/mnt/nvme_share/XXX/data/condition_positive_rates.csv', dtype=object)
        self.diagnosis_text = [diagnosis_csv.loc[i]['condition'] for i in range(len(diagnosis_csv))]
        class_instance_nums = [int(diagnosis_csv.loc[i]['count']) for i in range(len(diagnosis_csv))]
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float16)
        p = class_instance_nums / 69235
        self.pos_weights = torch.exp(1-p)
        self.neg_weights = torch.exp(p)
        
         
        
      
        
    def _register(self, llm_layers_index):
        def attention_hook(module, input, output):
            self.hidden.append(input[0])

        def output_hook(module, input, output):
            self.logit.append(output)

        self.text_decoder.base_model.model.model.layers[llm_layers_index].register_forward_hook(attention_hook)
        self.text_decoder.base_model.model.lm_head.register_forward_hook(output_hook)
        
        
        
    def forward(self, image, captions, cls_labels, criterion_cls, batch_idx):
        if image.dim()==6:
            b, num_seq, t, c, h, w = image.shape
            patch_feats = self.visual_encoder(image) # (b num_seq) (t v) d
            feat_size = patch_feats.shape[-1]
            patch_feats = self.resampler(patch_feats)
            image_embeds = patch_feats.reshape(b, -1 ,feat_size)
            avg_embeds = torch.mean(image_embeds, dim=1)
        else:
            image_embeds, _ = self.visual_encoder(image)
            image_embeds = self.resampler(image_embeds) # b,num_latents,2048
            avg_embeds = torch.mean(image_embeds, dim=1)

        cls_preds = self.cls_head(avg_embeds) # b, 2048
        cls_preds = cls_preds.view(-1, 2, self.args.num_label)
        cls_preds = self.logit_scale * cls_preds
        loss_cls = criterion_cls(cls_preds, cls_labels)
        diagnosis_weights = cls_labels * self.pos_weights.to(image.device) + (1 - cls_labels) * self.neg_weights.to(image.device)
        loss_cls = loss_cls*diagnosis_weights
        loss_cls = torch.mean(loss_cls.sum(dim=-1))


        #loss_cls = self.criterion_asl(cls_preds, cls_labels)


        if batch_idx%20==0:
            print(image_embeds[:3,50,:8])
            print('logit_scale: ', self.logit_scale)
            print(cls_preds[:3,:,:6])

        if not self.args.condition:

            captions = ['<|im_start|>请根据影像生成膝关节诊断报告。'+sample+'<|im_end|>' for sample in captions]
            caption_tokenization = self.tokenizer(captions, padding='longest', return_tensors="pt").to(image.device)
            text_embeds = self.text_decoder.get_input_embeddings()(caption_tokenization.input_ids)
            image_embeds = self.vision_proj(image_embeds)

            inputs_embeds = torch.cat([text_embeds[:,:1,:],image_embeds,text_embeds[:,1:,:]], dim=1) #bos,image_embeds,text_embeds,eos
            mask_paddings = torch.ones(image_embeds.shape[0], image_embeds.shape[1]).to(image.device)
            attention_mask = torch.cat([mask_paddings,caption_tokenization.attention_mask], dim=1)
            decoder_targets = caption_tokenization.input_ids.masked_fill(caption_tokenization.input_ids == self.tokenizer.pad_token_id, -100)
            prompt_len = len(self.tokenizer('<|im_start|>请根据影像生成膝关节诊断报告。').input_ids)
            decoder_targets[:,:prompt_len] = -100
            ignored_paddings = torch.full((image_embeds.shape[0], image_embeds.shape[1]), -100).to(image.device)
            decoder_targets = torch.cat([ignored_paddings, decoder_targets], dim=1)

        else: # self.args.condition==True
            prompt_labels = cls_labels.clone()
            prompt_lens = []
            captions_new = []
            for prompt_label, caption in zip(prompt_labels, captions):
                diagnosis = ''
                for idx, item in enumerate(prompt_label):
                    if item==1:
                        if self.args.mask_prompt: # diagnosis prompt masking
                            prob = random.random()
                            if prob < self.args.retain_ratio:
                                diagnosis = diagnosis + self.diagnosis_text[idx] + ';'
                            elif self.args.retain_ratio < prob < self.args.retain_ratio + self.args.drop_ratio:
                                pass
                            else:
                                idx = random.randint(0, self.args.num_label - 1)
                                diagnosis = diagnosis + self.diagnosis_text[idx] + ';'
                        else:
                            diagnosis = diagnosis + self.diagnosis_text[idx] + ';'
                prompt = '<|im_start|>' + diagnosis + '请根据以上信息生成膝关节诊断报告。'
                prompt_len = len(self.tokenizer(prompt).input_ids)
                prompt_lens.append(prompt_len)
                caption_new = prompt + caption + '<|im_end|>'
                captions_new.append(caption_new)

            caption_tokenization = self.tokenizer(captions_new, padding='longest', return_tensors="pt").to(image.device)
            text_embeds = self.text_decoder.get_input_embeddings()(caption_tokenization.input_ids)
            image_embeds = self.vision_proj(image_embeds)
            inputs_embeds = torch.cat([text_embeds[:,:1,:],image_embeds,text_embeds[:,1:,:]], dim=1) #bos,image_embeds,text_embeds,eos
            mask_paddings = torch.ones(image_embeds.shape[0], image_embeds.shape[1]).to(image.device)
            attention_mask = torch.cat([mask_paddings,caption_tokenization.attention_mask], dim=1)
            decoder_targets = caption_tokenization.input_ids.masked_fill(caption_tokenization.input_ids == self.tokenizer.pad_token_id, -100)
            # toy_mask = torch.arange(decoder_targets.shape[1]).repeat(decoder_targets.shape[0]).reshape(decoder_targets.shape[0], decoder_targets.shape[1]).to(image.device)
            # toy_mask_index = toy_mask<prompt_lens.view(decoder_targets.shape[0], -1)
            # decoder_targets[toy_mask_index] = -100
            for i, index in enumerate(prompt_lens):
                decoder_targets[i,:index] = -100
            ignored_paddings = torch.full((image_embeds.shape[0], image_embeds.shape[1]), -100).to(image.device)
            decoder_targets = torch.cat([ignored_paddings, decoder_targets], dim=1)



        decoder_output = self.text_decoder(attention_mask = attention_mask, 
                                           inputs_embeds = inputs_embeds, 
                                           labels = decoder_targets,
                                           return_dict = True,  
                                          )
        loss_lm = decoder_output.loss
            

        return loss_lm, loss_cls

        
    def generate(self, image, sample=False, num_beams=3, max_length=100, min_length=10, top_p=0.9, repetition_penalty=1.0, batch_idx=0):
        if image.dim()==6:
            b, num_seq, t, c, h, w = image.shape
            patch_feats = self.visual_encoder(image) # (b num_seq) (t v) d
            feat_size = patch_feats.shape[-1]
            patch_feats = self.resampler(patch_feats)
            image_embeds = patch_feats.reshape(b, -1 ,feat_size)
            avg_embeds = torch.mean(image_embeds, dim=1)
        else:
            image_embeds, _ = self.visual_encoder(image)
            image_embeds = self.resampler(image_embeds) # b,num_latents,2048
            avg_embeds = torch.mean(image_embeds, dim=1)
        
        cls_preds = self.cls_head(avg_embeds)
        cls_preds = cls_preds.view(-1, 2, self.args.num_label)
        cls_preds = self.logit_scale * cls_preds
        cls_preds = F.softmax(cls_preds, dim=1)
        cls_labels = (cls_preds[:,1,:]>=0.50).int()
       


        if not self.args.condition:

            prompts = ['<|im_start|>请根据影像生成膝关节诊断报告。']*image_embeds.shape[0]
            prompts_tokenization = self.tokenizer(prompts, padding='longest', return_tensors="pt").to(image.device)
            text_embeds = self.text_decoder.get_input_embeddings()(prompts_tokenization.input_ids)
            image_embeds = self.vision_proj(image_embeds)
            inputs_embeds = torch.cat([text_embeds[:,:1,:],image_embeds,text_embeds[:,1:,:]], dim=1)

            mask_paddings = torch.ones(image_embeds.shape[0], image_embeds.shape[1]).to(image.device)
            attention_mask = torch.cat([mask_paddings,prompts_tokenization.attention_mask], dim=1)

            outputs = self.text_decoder.generate(inputs_embeds=inputs_embeds,
                                                #min_length=min_length, # 4.25 Transformers
                                                max_new_tokens=256,
                                                num_beams=num_beams,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                #pad_token_id=self.tokenizer.pad_token_id, 
                                                #repetition_penalty=repetition_penalty,
                                                attention_mask = attention_mask,
                                                )
            captions = [] 
            for i, output in enumerate(outputs):
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption)
            if batch_idx%10==0:
                print(captions)
            return captions

        else:
            captions = []
            image_embeds = self.vision_proj(image_embeds)
            for idx_s, sample in enumerate(cls_labels):
                diagnosis_label = ''
                for idx, item in enumerate(sample):
                    if item==1:
                        diagnosis_label = diagnosis_label + self.diagnosis_text[idx] + ';'
                
                prompt = '<|im_start|>' + diagnosis_label + '请根据以上信息生成膝关节诊断报告。'
                prompt_tokenization = self.tokenizer(prompt, padding='longest', return_tensors="pt").to(image.device)
                text_embeds = self.text_decoder.get_input_embeddings()(prompt_tokenization.input_ids)
                inputs_embeds = torch.cat([text_embeds[:,:1,:],image_embeds[idx_s,:,:].unsqueeze(0),text_embeds[:,1:,:]], dim=1)

                mask_paddings = torch.ones(1, image_embeds.shape[1]).to(image.device)
                attention_mask = torch.cat([mask_paddings,prompt_tokenization.attention_mask], dim=1)

                outputs = self.text_decoder.generate(inputs_embeds=inputs_embeds,
                                                    #min_length=min_length, # 4.25 Transformers
                                                    max_new_tokens=256,
                                                    num_beams=num_beams,
                                                    eos_token_id=self.tokenizer.eos_token_id,
                                                    #pad_token_id=self.tokenizer.pad_token_id, 
                                                    #repetition_penalty=repetition_penalty,
                                                    attention_mask = attention_mask,
                                                    )
                caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                captions.append(caption)
            
            if batch_idx%20==0:
                print(captions)
            return captions


   
    
