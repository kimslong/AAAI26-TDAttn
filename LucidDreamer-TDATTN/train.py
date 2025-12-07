#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import yaml
import imageio
import matplotlib.pyplot as plt
from utils.loss_utils import l1_loss, ssim, tv_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GenerateCamParams, GuidanceParams
import math

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from tqdm.notebook import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
from pathlib import Path
import os
import random
from random import randint

logging.set_verbosity_error()
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd
from guidance.perpneg_utils import weighted_perpendicular_aggregator
from guidance.sd_step import *


def rgb2sat(img, T=None):
    max_ = torch.max(img, dim=1, keepdim=True).values + 1e-5
    min_ = torch.min(img, dim=1, keepdim=True).values
    sat = (max_ - min_) / max_
    if T is not None:
        sat = (1 - T) * sat
    return sat


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, t_range=[0.02, 0.98], max_t_range=0.98, num_train_timesteps=None,
                 ddim_inv=False, use_control_net=False, textual_inversion_path=None,
                 LoRA_path=None, guidance_opt=None):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32

        print(f'[INFO] loading stable diffusion...')

        model_key = guidance_opt.model_key
        assert model_key is not None

        is_safe_tensor = guidance_opt.is_safe_tensor
        base_model_key = "stabilityai/stable-diffusion-v1-5" if guidance_opt.base_model_key is None else guidance_opt.base_model_key  # for finetuned model only

        if is_safe_tensor:
            pipe = StableDiffusion.from_single_file(model_key, use_safetensors=True,
                                                    torch_dtype=self.precision_t, load_safety_checker=False)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        self.ism = not guidance_opt.sds
        self.scheduler = DDIMScheduler.from_pretrained(model_key if not is_safe_tensor else base_model_key,
                                                       subfolder="scheduler", torch_dtype=self.precision_t)
        self.sche_func = ddim_step

        if use_control_net:
            controlnet_model_key = guidance_opt.controlnet_model_key
            self.controlnet_depth = ControlNetModel.from_pretrained(controlnet_model_key,
                                                                    torch_dtype=self.precision_t).to(device)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        pipe.enable_xformers_memory_efficient_attention()

        pipe = pipe.to(self.device)
        if textual_inversion_path is not None:
            pipe.load_textual_inversion(textual_inversion_path)
            print("load textual inversion in:.{}".format(textual_inversion_path))

        if LoRA_path is not None:
            from lora_diffusion import tune_lora_scale, patch_pipe
            print("load lora in:.{}".format(LoRA_path))
            patch_pipe(
                pipe,
                LoRA_path,
                patch_text=True,
                patch_ti=True,
                patch_unet=True,
            )
            tune_lora_scale(pipe.unet, 1.00)
            tune_lora_scale(pipe.text_encoder, 1.00)

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.num_train_timesteps = num_train_timesteps if num_train_timesteps is not None else self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0,))
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.warmup_step = int(self.num_train_timesteps * (max_t_range - t_range[1]))

        self.noise_temp = None
        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(guidance_opt.noise_seed)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        self.rgb_latent_factors = torch.tensor([
            # R       G       B
            [0.298, 0.207, 0.208],
            [0.187, 0.286, 0.173],
            [-0.158, 0.189, 0.264],
            [-0.184, -0.271, -0.473]
        ], device=self.device)

        print(f'[INFO] loaded stable diffusion!')

    def augmentation(self, *tensors):
        augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
        ])

        channels = [ten.shape[1] for ten in tensors]
        tensors_concat = torch.concat(tensors, dim=1)
        tensors_concat = augs(tensors_concat)

        results = []
        cur_c = 0
        for i in range(len(channels)):
            results.append(tensors_concat[:, cur_c:cur_c + channels[i], ...])
            cur_c += channels[i]
        return (ten for ten in results)

    def add_noise_with_cfg(self, latents, noise,
                           ind_t, ind_prev_t,
                           text_embeddings=None, cfg=1.0,
                           delta_t=1, inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0):

        text_embeddings = text_embeddings.to(self.precision_t)
        if cfg <= 1.0:
            uncond_text_embedding = \
                text_embeddings.reshape(2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        unet = self.unet

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(
                self.precision_t)

            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0],
                                                                                      1).reshape(-1)
                unet_output = unet(latent_model_input, timestep_model_input,
                                   encoder_hidden_states=text_embeddings).sample

                uncond, cond = torch.chunk(unet_output, chunks=2)

                unet_output = cond + cfg * (uncond - cond)  # reverse cfg to enhance the distillation
            else:
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0],
                                                                                      1).reshape(-1)
                unet_output = unet(cur_noisy_lat_, timestep_model_input,
                                   encoder_hidden_states=uncond_text_embedding).sample

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t - cur_t if isinstance(self.scheduler, DDIMScheduler) else next_ind_t - cur_ind_t

            cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_,
                                           eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]

    @torch.no_grad()
    def get_text_embeds(self, prompt, resolution=(512, 512)):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def train_step_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                           grad_scale=1, use_control_net=False,
                           save_folder: Path = None, iteration=0, warm_up_rate=0, weights=0,
                           resolution=(512, 512), guidance_opt=None, as_latent=False, embedding_inverse=None,
                           render_attns0=None):

        def normalize_to_01(tensor):
            min_val = tensor.min()
            max_val = tensor.max()
            normalized = (tensor - min_val) / (max_val - min_val + 1e-6)
            return torch.clamp(normalized, 0, 1)

        render_attns = normalize_to_01(torch.stack(render_attns0, dim=0))

        # flip aug
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if as_latent:
            latents, _ = self.encode_imgs(pred_depth.repeat(1, 3, 1, 1).to(self.precision_t))
        else:
            latents, _ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level

        weights = weights.reshape(-1)
        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8,), dtype=latents.dtype,
                            device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1),
                                                                                                 device=latents.device).repeat(
            latents.shape[0], 1, 1, 1)

        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1,
                                                                                            embedding_inverse.shape[-2],
                                                                                            embedding_inverse.shape[-1])

        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2],
                                                  text_embeddings.shape[-1])  # make it k+1, c * t, ...

        if guidance_opt.annealing_intervals:
            current_delta_t = int(
                guidance_opt.delta_t + np.ceil((warm_up_rate) * (guidance_opt.delta_t_start - guidance_opt.delta_t)))
        else:
            current_delta_t = guidance_opt.delta_t

        ind_t = \
            torch.randint(self.min_step, self.max_step + int(self.warmup_step * warm_up_rate), (1,), dtype=torch.long,
                          generator=self.noise_gen, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        with torch.no_grad():
            # step unroll via ddim inversion
            if not self.ism:
                prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                target = noise
            else:
                # Step 1: sample x_s with larger steps
                xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
                xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(
                    np.ceil(ind_prev_t / xs_delta_t))
                starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

                standard_sd = True
                controller = TDATTN_SGT.AttentionReweight(num_steps=t, cross_replace_steps=[0.0, 0.0],
                                                          equalizers=equalizers,
                                                          adjusting_direction=['View', 'View'],
                                                          category_vectors=category_vectors_agrmax_2,
                                                          head_index=head_index,
                                                          standard_sd=standard_sd, skip_call=True, tokenizer=tokenizer,
                                                          SGT=SGT)
                tdattn.register_attention_control(guidance.pipe, controller)

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t,
                                                                                starting_ind, inverse_text_embeddings,
                                                                                guidance_opt.denoise_guidance_scale,
                                                                                xs_delta_t, xs_inv_steps,
                                                                                eta=guidance_opt.xs_eta)
                # Step 2: sample x_t
                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, ind_t, ind_prev_t,
                                                                           inverse_text_embeddings,
                                                                           guidance_opt.denoise_guidance_scale,
                                                                           current_delta_t, 1, is_noisy_latent=True)

                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]

        with torch.no_grad():
            latent_model_input = latents_noisy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4,
                                                                                               resolution[0] // 8,
                                                                                               resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
            if use_control_net:
                pred_depth_input = pred_depth_input[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, 512,
                                                                                                    512).half()
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    latent_model_input,
                    tt,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=pred_depth_input,
                    return_dict=False,
                )
                unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings,
                                        down_block_additional_residuals=down_block_res_samples,
                                        mid_block_additional_residual=mid_block_res_sample).sample
            else:

                if iteration <= 5000:
                    standard_sd = True
                    tdattn.standard_sd = True
                    if iteration<=2000:
                        if np.mean(np.abs(tdattn.azimuths)) > 100:
                            if random.random() < 0.3:
                                standard_sd = False
                                tdattn.standard_sd = False
                        else:
                            if random.random() < 0.1:
                                standard_sd = False
                                tdattn.standard_sd = False
                    controller = TDATTN_SGT.AttentionReweight(num_steps=t, cross_replace_steps=reweight_steps,
                                                              equalizers=equalizers,
                                                              adjusting_direction=adjusting_direction,
                                                              category_vectors=category_vectors_agrmax_2,
                                                              head_index=head_index,
                                                              standard_sd=standard_sd, batch_size=tdattn.c_batchsize,
                                                              tokenizer=tokenizer, SGT=SGT)
                    skip_call = False
                    tdattn.register_attention_control(guidance.pipe, controller)


                else:
                    standard_sd = True
                    skip_call = True
                    controller = TDATTN_SGT.AttentionReweight(num_steps=t, cross_replace_steps=[0.0, 0.0],
                                                              equalizers=equalizers,
                                                              adjusting_direction=adjusting_direction,
                                                              category_vectors=category_vectors_agrmax_2,
                                                              head_index=head_index,
                                                              standard_sd=standard_sd, skip_call=True,
                                                              batch_size=tdattn.c_batchsize, tokenizer=tokenizer,
                                                              SGT=SGT)

                    tdattn.register_attention_control(guidance.pipe, controller)

                unet_output = self.unet(latent_model_input.to(self.precision_t), tt.to(self.precision_t),
                                        encoder_hidden_states=text_embeddings.to(self.precision_t)).sample

            unet_output = unet_output.reshape(1 + K, -1, 4, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8,
                                                                         resolution[1] // 8, ), unet_output[1:].reshape(
                -1, 4, resolution[0] // 8, resolution[1] // 8, )
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            delta_DSD = weighted_perpendicular_aggregator(delta_noise_preds, \
                                                          weights, \
                                                          B)

        pred_noise = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)

        grad = w(self.alphas[t]) * (pred_noise - target)

        grad = torch.nan_to_num(grad_scale * grad)
        loss = SpecifyGradient.apply(latents, grad)

        show_flag = 0
        if controller is not None and not skip_call:
            # if iteration % 50 == 0:
            #     save_2d_attn = 0
            # else:
            #     save_2d_attn = 0
            # if iteration % 10 == 0:
            save_path_iter_attn = os.path.join(save_folder, "iter_{}_step_{}_attn".format(iteration, prev_t.item()))
            image_key32 = tdattn.show_cross_attention(guidance.tokenizer, save_path_iter_attn,
                                                      controller.attention_store, res=32,
                                                      from_where=("up", "down"), save_2d_attn=save_2d_attn,
                                                      no_key=no_key)
            del controller, delta_noise_preds, latent_model_input, noise_pred_text, target, text_embeddings, unet_output
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            if iteration <=5000:
                show_flag = 1

                def get_gauss_kernel(kernel_size: int, sigma: float, device):
                    ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device, dtype=torch.float32)
                    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
                    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
                    kernel = kernel / kernel.sum()
                    return kernel

                def gaussian_blur_tensor(x: torch.Tensor, kernel_size=15, sigma=3.0) -> torch.Tensor:
                    B, H, W = x.shape
                    kernel2d = get_gauss_kernel(kernel_size, sigma, x.device)
                    kernel2d = kernel2d.to(dtype=x.dtype)
                    kernel2d = kernel2d.view(1, 1, kernel_size, kernel_size)
                    x = x.unsqueeze(1)

                    x_blur = F.conv2d(x, weight=kernel2d, bias=None,
                                      stride=1, padding=kernel_size // 2, groups=1)  # groups=1 表示单通道卷积
                    return x_blur.squeeze(1)

                def adjust_brightness(attn_map: torch.Tensor, factor: float) -> torch.Tensor:
                    return torch.clamp(attn_map * factor, 0.0, 1.0)

                def kl_div_loss(p, q):
                    p = F.log_softmax(p.reshape(1, -1), dim=1)
                    q = F.softmax(q.reshape(1, -1), dim=1)
                    kl_loss = F.kl_div(p, q, reduction='batchmean')
                    return torch.clamp(kl_loss, min=0)

                if tdattn.c_batchsize == 1:
                    gray = image_key32[:, :, 0]
                    attn_2d = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).cuda()
                    tdattn.update_mask(tdattn.viewpoint_cams[0], attn_2d)
                else:
                    for i in range(tdattn.c_batchsize):
                        gray = image_key32[i]
                        attn_2d = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).cuda()
                        tdattn.update_mask(tdattn.viewpoint_cams[i], attn_2d)
                if iteration>200:
                    latents_blur = gaussian_blur_tensor(render_attns, kernel_size=31, sigma=2.0)
                    latents_blur_normalized = adjust_brightness(latents_blur, factor=1.5)
                    image_key32 = [torch.tensor(img) if isinstance(img, np.ndarray) else img for img in image_key32]
                    image_key32_stack = torch.stack(image_key32, dim=0).cuda()
                    image_key32_stack = normalize_to_01(image_key32_stack)
                    loss_kl = 0
                    for i in range(tdattn.c_batchsize):
                        loss_kl += kl_div_loss(latents_blur_normalized[i], image_key32_stack[i])
                    print(f"kl_loss={loss_kl},[ITER]={iteration}")
                    loss = loss + 10*loss_kl

        if iteration % 50 == 0:
            noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD
            lat2rgb = lambda x: torch.clip(
                (x.permute(0, 2, 3, 1) @ self.rgb_latent_factors.to(x.dtype)).permute(0, 3, 1, 2), 0., 1.)
            save_path_iter = os.path.join(save_folder, "iter_{}_step_{}.jpg".format(iteration, prev_t.item()))
            with torch.no_grad():
                pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))

                grad_abs = torch.abs(grad.detach())
                norm_grad = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1, keepdim=True),
                                          (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(
                    1, 3, 1, 1)

                latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear',
                                            align_corners=False)
                latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]),
                                               mode='bilinear', align_corners=False)

                if show_flag:
                    viz_images = torch.cat([pred_rgb,
                                            pred_depth.repeat(1, 3, 1, 1),
                                            pred_alpha.repeat(1, 3, 1, 1),
                                            rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                            latents_rgb, latents_sp_rgb,
                                            norm_grad,
                                            pred_x0_sp, pred_x0_pos,
                                            render_attns.unsqueeze(1).repeat(1, 3, 1, 1),
                                            image_key32_stack.unsqueeze(1).repeat(1, 3, 1, 1), ], dim=0)
                else:
                    viz_images = torch.cat([pred_rgb,
                                            pred_depth.repeat(1, 3, 1, 1),
                                            pred_alpha.repeat(1, 3, 1, 1),
                                            rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                            latents_rgb, latents_sp_rgb,
                                            norm_grad,
                                            pred_x0_sp, pred_x0_pos], dim=0)
                save_image(viz_images, save_path_iter)

        return loss

    def train_step(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                   grad_scale=1, use_control_net=False,
                   save_folder: Path = None, iteration=0, warm_up_rate=0,
                   resolution=(512, 512), guidance_opt=None, as_latent=False, embedding_inverse=None):

        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if as_latent:
            latents, _ = self.encode_imgs(pred_depth.repeat(1, 3, 1, 1).to(self.precision_t))
        else:
            latents, _ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level

        if self.noise_temp is None:
            self.noise_temp = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8,),
                                          dtype=latents.dtype, device=latents.device,
                                          generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1),
                                                                                        device=latents.device).repeat(
                latents.shape[0], 1, 1, 1)

        if guidance_opt.fix_noise:
            noise = self.noise_temp
        else:
            noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8,), dtype=latents.dtype,
                                device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1),
                                                                                                     device=latents.device).repeat(
                latents.shape[0], 1, 1, 1)

        text_embeddings = text_embeddings[:, :, ...]
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2],
                                                  text_embeddings.shape[-1])  # make it k+1, c * t, ...

        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1,
                                                                                            embedding_inverse.shape[-2],
                                                                                            embedding_inverse.shape[-1])

        if guidance_opt.annealing_intervals:
            current_delta_t = int(
                guidance_opt.delta_t + (warm_up_rate) * (guidance_opt.delta_t_start - guidance_opt.delta_t))
        else:
            current_delta_t = guidance_opt.delta_t

        ind_t = \
            torch.randint(self.min_step, self.max_step + int(self.warmup_step * warm_up_rate), (1,), dtype=torch.long,
                          generator=self.noise_gen, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        with torch.no_grad():
            # step unroll via ddim inversion
            if not self.ism:
                prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                target = noise
            else:
                # Step 1: sample x_s with larger steps
                xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
                xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(
                    np.ceil(ind_prev_t / xs_delta_t))
                starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t,
                                                                                starting_ind, inverse_text_embeddings,
                                                                                guidance_opt.denoise_guidance_scale,
                                                                                xs_delta_t, xs_inv_steps,
                                                                                eta=guidance_opt.xs_eta)
                # Step 2: sample x_t
                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, ind_t, ind_prev_t,
                                                                           inverse_text_embeddings,
                                                                           guidance_opt.denoise_guidance_scale,
                                                                           current_delta_t, 1, is_noisy_latent=True)

                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]

        with torch.no_grad():
            latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8,
                                                                                           resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
            if use_control_net:
                pred_depth_input = pred_depth_input[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, 512,
                                                                                                    512).half()
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    latent_model_input,
                    tt,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=pred_depth_input,
                    return_dict=False,
                )
                unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings,
                                        down_block_additional_residuals=down_block_res_samples,
                                        mid_block_additional_residual=mid_block_res_sample).sample
            else:
                unet_output = self.unet(latent_model_input.to(self.precision_t), tt.to(self.precision_t),
                                        encoder_hidden_states=text_embeddings.to(self.precision_t)).sample

            unet_output = unet_output.reshape(2, -1, 4, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8,
                                                                         resolution[1] // 8, ), unet_output[1:].reshape(
                -1, 4, resolution[0] // 8, resolution[1] // 8, )
            delta_DSD = noise_pred_text - noise_pred_uncond

        pred_noise = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD

        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)

        grad = w(self.alphas[t]) * (pred_noise - target)

        grad = torch.nan_to_num(grad_scale * grad)
        loss = SpecifyGradient.apply(latents, grad)

        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + 7.5 * delta_DSD
            lat2rgb = lambda x: torch.clip(
                (x.permute(0, 2, 3, 1) @ self.rgb_latent_factors.to(x.dtype)).permute(0, 3, 1, 2), 0., 1.)
            save_path_iter = os.path.join(save_folder, "iter_{}_step_{}.jpg".format(iteration, prev_t.item()))
            with torch.no_grad():
                pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))
                # pred_x0_uncond = pred_x0_sp[:1, ...]

                grad_abs = torch.abs(grad.detach())
                norm_grad = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1, keepdim=True),
                                          (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(
                    1, 3, 1, 1)

                latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear',
                                            align_corners=False)
                latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]),
                                               mode='bilinear', align_corners=False)

                viz_images = torch.cat([pred_rgb,
                                        pred_depth.repeat(1, 3, 1, 1),
                                        pred_alpha.repeat(1, 3, 1, 1),
                                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        latents_rgb, latents_sp_rgb, norm_grad,
                                        pred_x0_sp, pred_x0_pos], dim=0)
                save_image(viz_images, save_path_iter)

        return loss

    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs.to(target_dtype)

    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence


def adjust_text_embeddings(embeddings, azimuth, guidance_opt):
    # TODO: add prenerg functions
    text_z_list = []
    weights_list = []
    K = 0
    # for b in range(azimuth):
    text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth, guidance_opt)
    K = max(K, weights_.shape[0])
    text_z_list.append(text_z_)
    weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0)  # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0)  # [B * K]
    return text_embeddings, weights


def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1 - r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)


def prepare_embeddings(guidance_opt, guidance):
    embeddings = {}
    # text embeddings (stable-diffusion) and (IF)
    embeddings['default'] = guidance.get_text_embeds([guidance_opt.text])
    embeddings['uncond'] = guidance.get_text_embeds([guidance_opt.negative])

    for d in ['front', 'side', 'back']:
        embeddings[d] = guidance.get_text_embeds([f"{guidance_opt.text}, {d} view"])
    embeddings['inverse_text'] = guidance.get_text_embeds(guidance_opt.inverse_text)
    return embeddings


def guidance_setup(guidance_opt):
    if guidance_opt.guidance == "SD":
        # from guidance.sd_utils import StableDiffusion
        guidance = StableDiffusion(guidance_opt.g_device, guidance_opt.fp16, guidance_opt.vram_O,
                                   guidance_opt.t_range, guidance_opt.max_t_range,
                                   num_train_timesteps=guidance_opt.num_train_timesteps,
                                   ddim_inv=guidance_opt.ddim_inv,
                                   textual_inversion_path=guidance_opt.textual_inversion_path,
                                   LoRA_path=guidance_opt.LoRA_path,
                                   guidance_opt=guidance_opt)
    else:
        raise ValueError(f'{guidance_opt.guidance} not supported.')
    if guidance is not None:
        for p in guidance.parameters():
            p.requires_grad = False
    embeddings = prepare_embeddings(guidance_opt, guidance)
    return guidance, embeddings


def training(dataset, opt, pipe, gcams, guidance_opt, testing_iterations, saving_iterations, checkpoint_iterations,
             checkpoint, debug_from, save_video):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    scene = Scene(dataset, gcams, tdattn)
    tdattn.training_setup(opt)
    tdattn.GSweights = torch.zeros_like(
        tdattn._opacity)
    tdattn.GSweights_cnt = torch.zeros_like(tdattn._opacity,
                                            dtype=torch.int32)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        tdattn.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    save_folder = os.path.join(dataset._model_path, "train_process/")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs
        print('train_process is in :', save_folder)
    # controlnet
    use_control_net = False
    # set up pretrain diffusion models and text_embedings
    # guidance, embeddings = guidance_setup(guidance_opt)
    viewpoint_stack = None
    viewpoint_stack_around = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if opt.save_process:
        save_folder_proc = os.path.join(scene.args._model_path, "process_videos/")
        if not os.path.exists(save_folder_proc):
            os.makedirs(save_folder_proc)  # makedirs
        process_view_points = scene.getCircleVideoCameras(batch_size=opt.pro_frames_num,
                                                          render45=opt.pro_render_45).copy()
        save_process_iter = opt.iterations // len(process_view_points)
        pro_img_frames = []
    C_batch_size = guidance_opt.C_batch_size
    tdattn.c_batchsize = C_batch_size
    tdattn.prompts = [guidance_opt.text]
    for iteration in range(first_iter, opt.iterations + 1):
        # TODO: DEBUG NETWORK_GUI
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, tdattn, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, guidance_opt.text)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        tdattn.update_learning_rate(iteration)
        tdattn.update_feature_learning_rate(iteration)
        tdattn.update_rotation_learning_rate(iteration)
        tdattn.update_scaling_learning_rate(iteration)
        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            tdattn.oneupSHdegree()

        # progressively relaxing view range
        if not opt.use_progressive:
            if iteration >= opt.progressive_view_iter and iteration % opt.scale_up_cameras_iter == 0:
                scene.pose_args.fovy_range[0] = max(scene.pose_args.max_fovy_range[0],
                                                    scene.pose_args.fovy_range[0] * opt.fovy_scale_up_factor[0])
                scene.pose_args.fovy_range[1] = min(scene.pose_args.max_fovy_range[1],
                                                    scene.pose_args.fovy_range[1] * opt.fovy_scale_up_factor[1])

                scene.pose_args.radius_range[1] = max(scene.pose_args.max_radius_range[1],
                                                      scene.pose_args.radius_range[1] * opt.scale_up_factor)
                scene.pose_args.radius_range[0] = max(scene.pose_args.max_radius_range[0],
                                                      scene.pose_args.radius_range[0] * opt.scale_up_factor)

                scene.pose_args.theta_range[1] = min(scene.pose_args.max_theta_range[1],
                                                     scene.pose_args.theta_range[1] * opt.phi_scale_up_factor)
                scene.pose_args.theta_range[0] = max(scene.pose_args.max_theta_range[0],
                                                     scene.pose_args.theta_range[0] * 1 / opt.phi_scale_up_factor)

                # opt.reset_resnet_iter = max(500, opt.reset_resnet_iter // 1.25)
                scene.pose_args.phi_range[0] = max(scene.pose_args.max_phi_range[0],
                                                   scene.pose_args.phi_range[0] * opt.phi_scale_up_factor)
                scene.pose_args.phi_range[1] = min(scene.pose_args.max_phi_range[1],
                                                   scene.pose_args.phi_range[1] * opt.phi_scale_up_factor)

                print('scale up theta_range to:', scene.pose_args.theta_range)
                print('scale up radius_range to:', scene.pose_args.radius_range)
                print('scale up phi_range to:', scene.pose_args.phi_range)
                print('scale up fovy_range to:', scene.pose_args.fovy_range)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getRandTrainCameras().copy()

        C_batch_size = guidance_opt.C_batch_size
        viewpoint_cams = []
        images = []
        text_z_ = []
        weights_ = []
        depths = []
        alphas = []
        scales = []
        render_attns = []  # TDATTN
        text_z_inverse = torch.cat([embeddings['uncond'], embeddings['inverse_text']], dim=0)
        azimuths = []
        for i in range(C_batch_size):
            try:
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            except:
                viewpoint_stack = scene.getRandTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            # pred text_z
            azimuth = viewpoint_cam.delta_azimuth
            text_z = [embeddings['uncond']]

            if guidance_opt.perpneg:
                text_z_comp, weights = adjust_text_embeddings(embeddings, azimuth, guidance_opt)
                text_z.append(text_z_comp)
                weights_.append(weights)

            else:
                if azimuth >= -90 and azimuth < 90:
                    if azimuth >= 0:
                        r = 1 - azimuth / 90
                    else:
                        r = 1 + azimuth / 90
                    start_z = embeddings['front']
                    end_z = embeddings['side']
                else:
                    if azimuth >= 0:
                        r = 1 - (azimuth - 90) / 90
                    else:
                        r = 1 + (azimuth + 90) / 90
                    start_z = embeddings['side']
                    end_z = embeddings['back']
                text_z.append(r * start_z + (1 - r) * end_z)

            text_z = torch.cat(text_z, dim=0)
            text_z_.append(text_z)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, tdattn, pipe, background,
                                sh_deg_aug_ratio=dataset.sh_deg_aug_ratio,
                                bg_aug_ratio=dataset.bg_aug_ratio,
                                shs_aug_ratio=dataset.shs_aug_ratio,
                                scale_aug_ratio=dataset.scale_aug_ratio)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth, alpha = render_pkg["depth"], render_pkg["alpha"]
            render_attn = render_pkg.get("render_attn", None)[0, :, :]  # TDATTN
            render_attns.append(render_attn)  # TDATTN
            scales.append(render_pkg["scales"])
            images.append(image)
            depths.append(depth)
            alphas.append(alpha)
            viewpoint_cams.append(viewpoint_cam)
        tdattn.azimuths = azimuths
        images = torch.stack(images, dim=0)
        depths = torch.stack(depths, dim=0)
        alphas = torch.stack(alphas, dim=0)

        # Loss
        warm_up_rate = 1. - min(iteration / opt.warmup_iter, 1.)
        guidance_scale = guidance_opt.guidance_scale
        _aslatent = False
        if iteration < opt.geo_iter or random.random() < opt.as_latent_ratio:
            _aslatent = True
        if iteration > opt.use_control_net_iter and (random.random() < guidance_opt.controlnet_ratio):
            use_control_net = True
        if guidance_opt.perpneg:
            loss = guidance.train_step_perpneg(torch.stack(text_z_, dim=1), images,
                                               pred_depth=depths, pred_alpha=alphas,
                                               grad_scale=guidance_opt.lambda_guidance,
                                               use_control_net=use_control_net, save_folder=save_folder,
                                               iteration=iteration, warm_up_rate=warm_up_rate,
                                               weights=torch.stack(weights_, dim=1),
                                               resolution=(gcams.image_h, gcams.image_w),
                                               guidance_opt=guidance_opt, as_latent=_aslatent,
                                               embedding_inverse=text_z_inverse, render_attns0=render_attns)
        else:
            loss = guidance.train_step(torch.stack(text_z_, dim=1), images,
                                       pred_depth=depths, pred_alpha=alphas,
                                       grad_scale=guidance_opt.lambda_guidance,
                                       use_control_net=use_control_net, save_folder=save_folder, iteration=iteration,
                                       warm_up_rate=warm_up_rate,
                                       resolution=(gcams.image_h, gcams.image_w),
                                       guidance_opt=guidance_opt, as_latent=_aslatent, embedding_inverse=text_z_inverse)
            # raise ValueError(f'original version not supported.')
        scales = torch.stack(scales, dim=0)

        loss_scale = torch.mean(scales, dim=-1).mean()
        loss_tv = tv_loss(images) + tv_loss(depths)
        # loss_bin = torch.mean(torch.min(alphas - 0.0001, 1 - alphas))

        loss = loss + opt.lambda_tv * loss_tv + opt.lambda_scale * loss_scale  # opt.lambda_tv * loss_tv + opt.lambda_bin * loss_bin + opt.lambda_scale * loss_scale +
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if opt.save_process:
                if iteration % save_process_iter == 0 and len(process_view_points) > 0:
                    viewpoint_cam_p = process_view_points.pop(0)
                    render_p = render(viewpoint_cam_p, tdattn, pipe, background, test=True)
                    img_p = torch.clamp(render_p["render"], 0.0, 1.0)
                    img_p = img_p.detach().cpu().permute(1, 2, 0).numpy()
                    img_p = (img_p * 255).round().astype('uint8')
                    pro_img_frames.append(img_p)

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, iter_start.elapsed_time(iter_end), testing_iterations, scene, render,
                            (pipe, background))
            if (iteration in testing_iterations):
                if save_video:
                    video_inference(iteration, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                tdattn.max_radii2D[visibility_filter] = torch.max(tdattn.max_radii2D[visibility_filter],
                                                                  radii[visibility_filter])
                tdattn.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    tdattn.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0:  # or (dataset._white_background and iteration == opt.densify_from_iter)
                    tdattn.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                tdattn.optimizer.step()
                tdattn.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((tdattn.capture(tdattn.GSweights, tdattn.GSweights_cnt), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if opt.save_process:
        imageio.mimwrite(os.path.join(save_folder_proc, "video_rgb.mp4"), pro_img_frames, fps=30, quality=8)


def prepare_output_and_logger(args):
    if not args._model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args._model_path = os.path.join("./output/", args.workspace)

    # Set up output folder
    print("Output folder: {}".format(args._model_path))
    os.makedirs(args._model_path, exist_ok=True)

    # copy configs
    if args.opt_path is not None:
        os.system(' '.join(['cp', args.opt_path, os.path.join(args._model_path, 'config.yaml')]))

    with open(os.path.join(args._model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args._model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        save_folder = os.path.join(scene.args._model_path, "test_six_views/{}_iteration".format(iteration))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)  # makedirs 创建文件时如果路径不存在会创建这个路径
            print('test views is in :', save_folder)
        torch.cuda.empty_cache()
        config = ({'name': 'test', 'cameras': scene.getTestCameras()})
        if config['cameras'] and len(config['cameras']) > 0:
            for idx, viewpoint in enumerate(config['cameras']):
                render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
                rgb, depth = render_out["render"], render_out["depth"]
                if depth is not None:
                    depth_norm = depth / depth.max()
                    save_image(depth_norm, os.path.join(save_folder, "render_depth_{}.png".format(viewpoint.uid)))

                image = torch.clamp(rgb, 0.0, 1.0)
                save_image(image, os.path.join(save_folder, "render_view_{}.png".format(viewpoint.uid)))
                if tb_writer:
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.uid), image[None],
                                         global_step=iteration)
            print("\n[ITER {}] Eval Done!".format(iteration))
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def video_inference(iteration, scene: Scene, renderFunc, renderArgs):
    sharp = T.RandomAdjustSharpness(3, p=1.0)

    save_folder = os.path.join(scene.args._model_path, "videos/{}_iteration".format(iteration))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs
        print('videos is in :', save_folder)
    torch.cuda.empty_cache()
    config = ({'name': 'test', 'cameras': scene.getCircleVideoCameras()})
    if config['cameras'] and len(config['cameras']) > 0:
        img_frames = []
        depth_frames = []
        print("Generating Video using", len(config['cameras']), "different view points")
        for idx, viewpoint in enumerate(config['cameras']):
            render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
            rgb, depth = render_out["render"], render_out["depth"]
            if depth is not None:
                depth_norm = depth / depth.max()
                depths = torch.clamp(depth_norm, 0.0, 1.0)
                depths = depths.detach().cpu().permute(1, 2, 0).numpy()
                depths = (depths * 255).round().astype('uint8')
                depth_frames.append(depths)

            image = torch.clamp(rgb, 0.0, 1.0)
            image = image.detach().cpu().permute(1, 2, 0).numpy()
            image = (image * 255).round().astype('uint8')
            img_frames.append(image)
            # save_image(image,os.path.join(save_folder,"lora_view_{}.jpg".format(viewpoint.uid)))
        # Img to Numpy
        imageio.mimwrite(os.path.join(save_folder, "video_rgb_{}.mp4".format(iteration)), img_frames, fps=30, quality=8)
        if len(depth_frames) > 0:
            imageio.mimwrite(os.path.join(save_folder, "video_depth_{}.mp4".format(iteration)), depth_frames, fps=30,
                             quality=8)
        print("\n[ITER {}] Video Save Done!".format(iteration))
    torch.cuda.empty_cache()


import yaml
parser = ArgumentParser(description="Training script parameters")

parser.add_argument('--opt', type=str, default=None)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6009)
parser.add_argument('--debug_from', type=int, default=-1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument("--test_ratio", type=int, default=2) # [2500,5000,7500,10000,12000]
parser.add_argument("--save_ratio", type=int, default=2) # [10000,12000]
parser.add_argument("--save_video", type=bool, default=False)
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default = None)
# parser.add_argument("--device", type=str, default='cuda')


args = Args()
lp = ModelParams(args)
op = OptimizationParams(args)
pp = PipelineParams(args)
gcp = GenerateCamParams(args)
gp = GuidanceParams(args)

if args.opt is not None:
    with open(args.opt) as f:
        opts = yaml.load(f, Loader=yaml.FullLoader)
    lp.load_yaml(opts.get('ModelParams', None))
    op.load_yaml(opts.get('OptimizationParams', None))
    pp.load_yaml(opts.get('PipelineParams', None))
    gcp.load_yaml(opts.get('GenerateCamParams', None))
    gp.load_yaml(opts.get('GuidanceParams', None))

    lp.opt_path = args.opt
    args.port = opts['port']
    args.save_video = opts.get('save_video', True)
    args.seed = opts.get('seed', 0)
    args.device = opts.get('device', 'cuda')

    # override device
    gp.g_device = args.device
    lp.data_device = args.device
    gcp.device = args.device

# save iterations
test_iter = [1] + [k * op.iterations // args.test_ratio for k in range(1, args.test_ratio)] + [op.iterations]
args.test_iterations = test_iter

save_iter = [k * op.iterations // args.save_ratio for k in range(1, args.save_ratio)] + [op.iterations]
args.save_iterations = save_iter

if args.start_checkpoint == None:
    checkpoint = None
else:
    checkpoint = f"./output/{lp.workspace}/chkpnt{args.start_checkpoint}.pth"

print('Test iter:', args.test_iterations)
print('Save iter:', args.save_iterations)
print("Optimizing " + lp._model_path)

torch.autograd.set_detect_anomaly(args.detect_anomaly)
guidance, embeddings = guidance_setup(gp)
# ***************************************************ADD-TDATTN********************************************************************#
import TDATTN_SGT
import numpy as np
from TDATTN_SGT import tdattn_sgt
import csv

tokenizer = guidance.tokenizer
reweight_steps = [0.0, 1.01]
category_vectors_group_norm = np.load("./HAB_scores/category_vectors_group_norm.npy")
category_vectors_group_agrmax = np.load("./HAB_scores/category_vectors_group_agrmax.npy")
category_vectors = np.load("./HAB_scores/category_vectors.npy")
head_index = np.load("./HAB_scores/head_index.npy")
category_vectors_group_norm_1 = category_vectors_group_norm / np.linalg.norm(category_vectors_group_norm, axis=0,
                                                                             keepdims=True)
category_vectors_group_agrmax_1 = category_vectors_group_agrmax / np.linalg.norm(category_vectors_group_agrmax,
                                                                                 axis=0,
                                                                                 keepdims=True)
category_vectors_norm_2 = category_vectors_group_norm_1 * category_vectors
category_vectors_agrmax_2 = category_vectors_group_agrmax_1 * category_vectors
LOW_RESOURCE = False
tdattn = tdattn_sgt(sh_degree=0)
data = []
description_file_path = "./SGT_list.csv"
with open(description_file_path, "r") as f:
    render_csv = csv.reader(f)
    for row in render_csv:
        data.append(row)
num_list = np.arange(1, 11).astype(str)
data = [row for row in data if row != []]
data = [row[0] for row in data if not any(num in row[0] for num in num_list)]
data[0] = data[0].replace("\ufeff", "")
SGT = data
model_version = "sd_v2_1_base"
prompts = [gp.text]
selected_token = gcp.selected_token
desired_concept = gcp.desired_concept
undesired_concept = gcp.undesired_concept
no_key = gcp.no_key
adjusting_direction = [desired_concept, undesired_concept]
equalizers = []
zero_prompt_equalizer = TDATTN_SGT.get_equalizer(prompts[0], (selected_token), (0,), tokenizer=tokenizer).cuda()
equalizers.append(zero_prompt_equalizer)
equalizers.append(1 - zero_prompt_equalizer)

# *************************************************************************************************************************#
training(lp, op, pp, gcp, gp, args.test_iterations, args.save_iterations, args.checkpoint_iterations,
         args.start_checkpoint, args.debug_from, args.save_video)

# All done
print("\nTraining complete.")
