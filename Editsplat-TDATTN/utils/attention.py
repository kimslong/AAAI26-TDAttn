import torch
from diffusers.models.attention import Attention
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.transform import resize

class MyCrossAttnProcessor:
    def __init__(self, layer_name):
        self.layer_name = layer_name

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        attn.sequence_length = sequence_length
        attn.batch_size = batch_size
        attn.token_length = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else sequence_length

        # print("batch_size: ", batch_size)
        # print("sequence_length: ", sequence_length)
        # print("num_heads: ", attn.heads)

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        # print("query.shape: ", query.shape)
        # print("key.shape: ", key.shape)
        # print("value.shape: ", value.shape)

        attention_probs = attn.get_attention_scores(query, key, attention_mask) # [batch_size * num_heads, sequence_length, token_length]

        # save attention map
        if not hasattr(attn, 'attn_maps'):
            attn.attn_maps = {}
        if self.layer_name not in attn.attn_maps:
            attn.attn_maps[self.layer_name] = []

        attn.attn_maps[self.layer_name].append(attention_probs)
        # print(f"attn.attn_probs.shape for {self.layer_name}: ", attention_probs.shape)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def prep_unet(unet, is_cross_attn=True):
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and 'attn2' in name: # cross attention layer만 고려
        # if isinstance(module, Attention) and 'attn2' in name and "up" in name: # only upsample layer
            processor = MyCrossAttnProcessor(name)
            module.set_processor(processor)
            # print(f"Set processor for {name}")

    return unet

def get_all_attention_maps(unet): 
    # return {layer_name: [attention_map1, attention_map2, ...]}
    attention_maps = {}
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and hasattr(module, 'attn_maps'):
            for layer_name, maps in module.attn_maps.items():
                if layer_name not in attention_maps:
                    attention_maps[layer_name] = []
                attention_maps[layer_name].extend(maps)
    return attention_maps

def reset_attention_maps(unet): # clear attention maps cache
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and hasattr(module, 'attn_maps'):
            module.attn_maps = {}

# for debugging
def print_attention_maps_info(attention_maps):
    for layer_name, maps in attention_maps.items():
        print(f"Key: {layer_name}")
        for idx, attn_map in enumerate(maps):
            print(f"Value {idx}: shape = {attn_map.shape}")

def seperate_attention_maps_by_tokens(unet, attention_maps, tokenizer, prompt):
    text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt") #{'<|startoftext|>_0': [], 'there</w>_1': [], 'is</w>_2': [], 'a</w>_3': [], 'panda</w>_4': [], '<|endoftext|>_5': [], '<|endoftext|>_6': [],,,}
    token_ids = text_inputs.input_ids[0]  # use the first batch
    attention_mask = text_inputs.attention_mask[0]  
    num_tokens = attention_mask.sum().item() # number of valid tokens

    # map valid token to index
    valid_token_ids = token_ids[:num_tokens]
    tokens = tokenizer.convert_ids_to_tokens(valid_token_ids)
    token_attention_maps = {f"{token}_{i}": [] for i, token in enumerate(tokens)}

    # get batch_size, num_heads, token_length
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and 'attn2' in name:
            batch_size = module.batch_size
            num_heads = module.heads
            token_length = module.token_length
            break

    for layer_name, maps in attention_maps.items():
        for attn_map in maps:
            # calculate average attention map for each head
            attn_map = attn_map.reshape(batch_size, num_heads, -1, token_length)
            avg_attn_map = attn_map.mean(axis=1)  

            for i, token_id in enumerate(valid_token_ids):
                token = tokenizer.convert_ids_to_tokens([token_id])[0]
                key = f"{token}_{i}"
                if key not in token_attention_maps:
                    token_attention_maps[key] = []
                token_attention_maps[key].append(avg_attn_map[:, :, i].mean(axis=0).cpu().detach().float().numpy())

    return token_attention_maps

def save_attention_maps(token_attention_maps, src_trg_attention_map, object_word=None, output_dir=None, image_height=512, image_width=512):
    object_average_attention_map = None
    object_average_attention_map_resized = None

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    src_trg_attention_map = list(src_trg_attention_map.keys())

    for token, attention_values in token_attention_maps.items():
        resized_attention_maps = []
        if len(attention_values) == 1:
            axes = [axes]
        for i, attention in enumerate(attention_values):
            # reshape attention map to 2D
            # if attention.shape[0] == 4096:
            #     attention = attention.reshape(64, 64)
            if attention.shape[0] == 1024:
                attention = attention.reshape(32, 32)
            # elif attention.shape[0] == 256:
            #     attention = attention.reshape(16, 16)
            # elif attention.shape[0] == 64:
            #     attention = attention.reshape(8, 8)

            resized_attention_maps.append(resize(attention, (64, 64)))

        # save average attention map
        average_attention_map = np.mean(resized_attention_maps, axis=0)
        # if output_dir is not None:
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(average_attention_map, cmap="viridis", aspect="equal")
        #     plt.savefig(os.path.join(output_dir, f"{token.replace('/', '_')}_average.png"))
        #     plt.close()

        # resize to 512
        average_attention_map = np.mean(resized_attention_maps, axis=0)
        average_attention_map_resized = resize(average_attention_map, (image_height, image_width))


        # if output_dir is not None:
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(average_attention_map_resized, cmap="viridis", aspect="equal")
        #     plt.title(f"Average Attention Map\n Token: {token}")
        #     plt.xlabel("Position")
        #     plt.ylabel("Attention Score")
        #     plt.tight_layout()
        #     # plt.savefig(os.path.join(output_dir, f"{token.replace('/', '_')}_average_resize_512.png"))
        #     plt.close()

        # print(token)
        if object_word in token:
            object_average_attention_map =  average_attention_map # 64x64
            object_average_attention_map_resized = average_attention_map_resized # 512x512

    return object_average_attention_map, object_average_attention_map_resized