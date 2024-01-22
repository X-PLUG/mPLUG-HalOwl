#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class LlavaConfig(LlamaConfig):
    model_type = "llava"

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, offset: int = 0):
    cos = cos[..., offset : x.shape[-2] + offset, :]
    sin = sin[..., offset : x.shape[-2] + offset, :]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlavaLlamaModel(LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None, queue_size=4096):
        super(LlavaLlamaModel, self).__init__(config)
        self.queue_size = queue_size
        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP
            self.vision_tower = CLIPVisionModel.from_pretrained(config.mm_vision_tower)
            # self.vision_tower = CLIPVisionModel.from_pretrained(config.mm_vision_tower)

            # self.vision_tower = CLIPVisionModel.from_pretrained(config.mm_vision_tower)
        
        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            

    def initialize_vision_modules(self, vision_tower, mm_vision_select_layer, pretrain_vision_tower=None,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        self.config.mm_vision_tower = vision_tower
        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        
        if not hasattr(self, 'vision_tower'):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower
        
        if pretrain_vision_tower is None:
            vision_tower.requires_grad_(True)
            print('----------vision_tower.requires_grad_(True)-------')
        else:
            vision_tower.requires_grad_(False)
            print('----------vision_tower.requires_grad_(False)-------')
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = vision_tower

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)
            self.rotary_emb=LlamaRotaryEmbedding(self.config.hidden_size)
            self.image_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)
            self.text_projector = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.register_buffer("image_queue", torch.randn(self.config.hidden_size, self.queue_size))
            self.register_buffer("text_queue", torch.randn(self.config.hidden_size, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
            self.itc_temp = nn.Parameter(torch.ones([]) * 0.07)
    
        if pretrain_mm_mlp_adapter is not None:     
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})
        
        if pretrain_vision_tower is not None:
            vision_tower_weigths =  torch.load(pretrain_vision_tower, map_location='cpu')
            self.vision_tower.load_state_dict({k[len('model.vision_tower.'):]: v for k, v in vision_tower_weigths.items()})
        
        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        caption_ids : torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data
        do_itc=False
        if inputs_embeds is None:
           
            inputs_embeds = self.embed_tokens(input_ids)
        if caption_ids is not None:
            do_itc=True
            caption_embeds = self.embed_tokens(caption_ids)
            cos, sin = self.rotary_emb(caption_embeds, seq_len=caption_embeds.shape[-2])
            caption_embeds = apply_rotary_pos_emb(caption_embeds, cos, sin, offset=0)
            text_global_embedding= F.normalize(self.text_projector(torch.mean(caption_embeds, dim = -2)),dim=-1).squeeze(0)
            
            # print('text_global_embedding.shape:',text_global_embedding.shape, 'torch.mean(caption_embeds, dim = 1).shape:',torch.mean(caption_embeds, dim = 1).shape,'caption_embeds.shape:',caption_embeds.shape)
        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            # vision_tower = vision_tower[0]  # HACK: for FSDP
            # with torch.no_grad():
            if type(images) is list:
                # variable length images
                image_features = []
                image_global_embedding=[]
                for image in images:
                    image_forward_out = self.vision_tower(image.unsqueeze(0), output_hidden_states=True)
                    select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                    select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                    image_feature = select_hidden_state[:, 1:]
                    image_features.append(image_feature)
                    image_global_embedding.append(select_hidden_state[:, :1])
            else:
                image_forward_outs =self.vision_tower(images, output_hidden_states=True)
                select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                image_features = select_hidden_state[:, 1:]
                image_global_embedding = select_hidden_state[:,0]
            
            if type(images) is list:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
                if do_itc:
                    image_global_embedding = [F.normalize(self.image_projector(image_global),dim=-1) for image_global in image_global_embedding]
            else:
                image_features = self.mm_projector(image_features)
                print(image_features.shape, torch.mean(image_features, dim = -2).shape)
                if do_itc:
                    image_global_embedding = F.normalize(self.image_projector(image_global_embedding),dim=-1)
            if do_itc:
                
                text_feat_all = torch.cat([text_global_embedding.t(), self.text_queue.clone().detach()], dim=1)
                image_feat_all = torch.cat([image_global_embedding.t(), self.image_queue.clone().detach()], dim=1)
                with torch.no_grad():
                    self.itc_temp.clamp_(0.001, 0.5)
                sim_i2t = image_global_embedding @ text_feat_all / self.itc_temp
                sim_t2i = text_global_embedding @ image_feat_all / self.itc_temp
                
                sim_targets = torch.zeros(sim_i2t.size()).to(inputs_embeds.device)
                sim_targets.fill_diagonal_(1)

                loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
                loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

                loss_ita = (loss_i2t + loss_t2i) / 2
            else:
                loss_ita = None
            dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)
            
            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_image_idx += 1
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (cur_input_ids == vision_tower.config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_image_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
        
        
        return super(LlavaLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        ),loss_ita

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

class LlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, itc_loss_log_file=None):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        
            
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
  
        # Initialize weights and apply final processing
        self.post_init()
        
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        caption_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs,loss_ita = self.model(
            input_ids=input_ids,
            caption_ids=caption_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if loss_ita is not None:
            # loss_dict = {'loss_itc':loss_ita.item().detach().cpu().numpy(), 'loss_normal':loss.item().detach().cpu().numpy()}
            # json.dump(loss_dict,self.itc_loss_log_file)
            # self.itc_loss_log_fil.writelines('\n')
            loss += loss_ita 
          
        if not return_dict:
           
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        # if loss_ita is not None:
        #     return CausalLMOutputWithPast(
        #         loss=loss,
        #         logits=logits,
        #         past_key_values=outputs.past_key_values,
        #         hidden_states=outputs.hidden_states,
        #         attentions=outputs.attentions,
        #     ), {'loss_ita':loss_ita, 'loss_fc':loss_fc }
        # else:
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        # vision_config = self.get_model().vision_tower[0].config
        vision_config = self.get_model().vision_tower.config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
