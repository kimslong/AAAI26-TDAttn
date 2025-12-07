import sys
import os
import random
from argparse import ArgumentParser
import json
import shutil
from jaxtyping import Float

from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import lpips

from diffusers import StableDiffusionInstructPix2PixPipeline, DDIMScheduler
import ImageReward as RM
from lang_sam import LangSAM

from scene import Scene, GaussianModel
from gaussian_renderer import render
from render import render_sets
from arguments import ModelParams, PipelineParams, OptimizationParams, EditingParams
from scene.dataloader import CameraDataset

from utils.attention import prep_unet, get_all_attention_maps, reset_attention_maps, seperate_attention_maps_by_tokens, \
    save_attention_maps
from utils.loss_utils import l1_loss
from utils.rgbd_warping import reproject_rgbd, reprojected2img
from utils.camera_proximity_utils import find_nearby_camera

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class editsplat_Pipeline(StableDiffusionInstructPix2PixPipeline):

    def encode_image(self, img_tensor: Float[torch.Tensor, "B C H W"], is_sample=False):

        vae_magic = 0.18215

        x = img_tensor
        x = 2 * x - 1
        x = x.float()

        if is_sample:
            # return varying latent variable for same input
            return self.vae.encode(x.to(self.weights_dtype)).latent_dist.sample() * vae_magic
        else:
            # return same latent variable for same input
            return self.vae.encode(x.to(self.weights_dtype)).latent_dist.mean * vae_magic

    def prepare_image_latents(self, imgs: Float[torch.Tensor, "BS 3 H W"]) -> Float[torch.Tensor, "BS 4 H W"]:

        imgs = 2 * imgs - 1
        image_latents = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist.mean

        uncond_image_latents = torch.zeros_like(image_latents).to(self.weights_dtype)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def latents_to_img(self, latents: Float[torch.Tensor, "BS 4 H W"]) -> Float[torch.Tensor, "BS 3 H W"]:

        vae_magic = 0.18215
        latents = 1 / vae_magic * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents.to(self.weights_dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def edit_image(
            self,
            text_embeddings: Float[torch.Tensor, "N max_length embed_dim"],
            image: Float[torch.Tensor, "BS C H W"],
            image_cond: Float[torch.Tensor, "BS C H W"],
            text_guidance_scale: float = 7.5,
            image_guidance_scale: float = 1.5,
            diffusion_steps: int = 20,
            lower_bound: float = 0.70,
            upper_bound: float = 0.98,
            step: int = 0
    ) -> torch.FloatTensor:

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self._execution_device)

        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        latents = self.encode_image(image)
        image_cond_latents = self.prepare_image_latents(image_cond)  # torch.Size([3, 4, 64, 64])

        # add noise
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])
        standard_sd = False
        controller = TDATTN_SGT.AttentionReweight(prompts, diffusion_steps, cross_replace_steps=reweight_steps,
                                                  equalizers=equalizers, adjusting_direction=adjusting_direction,
                                                  category_vectors=category_vectors,
                                                  head_index=head_index, standard_sd=standard_sd,
                                                  tokenizer=tokenizer, SGT=SGT)
        tdattn.register_attention_control(pipeline, controller)
        for i, t in enumerate(self.scheduler.timesteps):
            # with torch.no_grad():
            latent_model_input = torch.cat([latents] * 3)  # torch.Size([3, 4, 64, 64])
            latent_model_input = torch.cat([latent_model_input, image_cond_latents],
                                           dim=1)  # torch.Size([3, 8, 64, 64])

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                    noise_pred_uncond
                    + text_guidance_scale * (noise_pred_text - noise_pred_image)
                    + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )
            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to get edited image
        decoded_img = self.latents_to_img(latents)

        return decoded_img

    @torch.no_grad()
    def edit_image_MFG(
            self,
            text_embeddings: Float[torch.Tensor, "N max_length embed_dim"],
            image: Float[torch.Tensor, "BS C H W"],
            image_cond: Float[torch.Tensor, "BS C H W"],
            MF_image_cond: Float[torch.Tensor, "BS C H W"],
            text_guidance_scale: float = 7.5,
            source_guidance_scale: float = 0.5,
            MFG_scale: float = 1.0,
            diffusion_steps: int = 20,
            lower_bound: float = 0.70,
            upper_bound: float = 0.98,
            step: int = 0,
            prompt: str = ""
    ) -> torch.FloatTensor:

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self._execution_device)

        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        latents = self.encode_image(image)
        image_cond_latents = self.prepare_image_latents(image_cond)  # torch.Size([3, 4, 64, 64])
        MF_image_cond_latents = self.prepare_image_latents(MF_image_cond)  # torch.Size([3, 4, 64, 64])

        # add noise
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])
        for i, t in enumerate(self.scheduler.timesteps):
            # with torch.no_grad():
            latent_model_input = torch.cat([latents] * 3)  # torch.Size([3, 4, 64, 64])
            latent_model_input = torch.cat([latent_model_input, image_cond_latents],
                                           dim=1)  # torch.Size([3, 8, 64, 64])
            standard_sd = False
            controller = TDATTN_SGT.AttentionReweight(prompts, diffusion_steps, cross_replace_steps=reweight_steps,
                                                      equalizers=equalizers, adjusting_direction=adjusting_direction,
                                                      category_vectors=category_vectors,
                                                      head_index=head_index, standard_sd=standard_sd,
                                                      tokenizer=tokenizer, SGT=SGT)
            tdattn.register_attention_control(pipeline, controller)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform MFG (Multi-View Fusion Guidance)
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)

            latent_model_input_mf = torch.cat([latents] * 3)  # torch.Size([3, 4, 64, 64])
            latent_model_input_mf = torch.cat([latent_model_input_mf, MF_image_cond_latents],
                                              dim=1)  # torch.Size([3, 8, 64, 64])

            standard_sd = False
            controller = TDATTN_SGT.AttentionReweight(prompts, diffusion_steps, cross_replace_steps=reweight_steps,
                                                      equalizers=equalizers, adjusting_direction=adjusting_direction,
                                                      category_vectors=category_vectors,
                                                      head_index=head_index, standard_sd=standard_sd,
                                                      tokenizer=tokenizer, SGT=SGT,skip_call=True)
            tdattn.register_attention_control(pipeline, controller)
            noise_pred_mf = self.unet(latent_model_input_mf, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_text_mf, noise_pred_image_mf, noise_pred_uncond_mf = noise_pred_mf.chunk(3)

            noise_pred = (
                    noise_pred_uncond
                    + text_guidance_scale * (noise_pred_text_mf - noise_pred_image_mf)
                    + source_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    + MFG_scale * (noise_pred_image_mf - noise_pred_uncond_mf)
            )
            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to get edited image
        decoded_img = self.latents_to_img(latents)
        return decoded_img


    def __call__(
            self,
            dataset=None,
            opt=None,
            pipe=None,
            ed=None,
    ):

        self.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler",
                                                       torch_dtype=torch.bfloat16)
        self.num_train_timesteps = 1000
        self.alphas = self.scheduler.alphas_cumprod.to(self._execution_device)

        # set unet to save cross-attention map
        self.unet = prep_unet(self.unet)
        self.unet.eval()
        self.unet.requires_grad_(False)

        # set weights dtype to bfloat16
        self.weights_dtype = torch.bfloat16
        self.unet = self.unet.to(self.weights_dtype)

        # encode target prompt
        trg_prompt_embeds = self._encode_prompt(
            ed.target_prompt, device=self._execution_device, num_images_per_prompt=1, do_classifier_free_guidance=True,
            negative_prompt=""
        )

        # load ImageReward
        reward_model = RM.load("ImageReward-v1.0")

        # load Lang-SAM
        lang_sam = LangSAM()

        # load 3D Gaussian Splatting
        gaussians = GaussianModel(dataset.sh_degree)

        scene = Scene(dataset, gaussians)

        gaussians.training_setup(opt)

        if dataset.source_checkpoint:
            (model_params, first_iter) = torch.load(dataset.source_checkpoint)
            gaussians.restore(model_params, opt)
            start_iteration = first_iter

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=self._execution_device)

        # for multi-view attention weighting
        attn_list = []

        # utility setting
        topilimage = ToPILImage()

        # LPIPS Loss
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(self._execution_device)
        lpips_loss_fn.requires_grad_(False)

        # Get the training dataset
        train_dataset = CameraDataset(scene)

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=dataset.view_shuffling, num_workers=0
        )

        # Get Camera distance matrix
        camera_list = train_dataset.camera_list
        camera_dist_order, _ = find_nearby_camera(camera_list)

        image_height = camera_list[0].image_height
        image_width = camera_list[0].image_width
        from PIL import Image
        import torchvision.transforms as transforms
        # Initially edit all images
        with torch.no_grad():

            edited_image_list = []
            edited_image_pil_list_RM = []
            rendered_depth_list = []
            is_top_selection = []

            for step, batch in enumerate(tqdm(train_dataloader, desc="Initial editing progress")):

                gt_image = batch['gt_image'].to(self._execution_device)
                idx = batch['idx'].item()

                reset_attention_maps(self.unet)

                if gt_image.shape[2] != 512 or gt_image.shape[3] != 512:
                    gt_image = F.interpolate(gt_image, size=(512, 512), mode='bilinear', align_corners=True)

                edited_image = self.edit_image(  # torch.Size([1, 3, 512, 512])
                    trg_prompt_embeds,
                    gt_image,
                    gt_image,
                    text_guidance_scale=ed.text_guidance_scale,
                    image_guidance_scale=ed.image_guidance_scale,
                    diffusion_steps=20,
                    lower_bound=0.70,
                    upper_bound=0.98,
                    step = step
                )


                # Save edited image to list
                edited_image = F.interpolate(edited_image.float(), size=(image_height, image_width), mode='bilinear',
                                             align_corners=False).to(edited_image.dtype)

                edited_image_list.append(edited_image.squeeze(0).detach().cpu().clone())

                # save pil image for imagereward sampling
                edited_pil = topilimage(edited_image.squeeze(0))
                edited_image_pil_list_RM.append(edited_pil)

                # render depth map from pretrained 3dgs
                render_pkg = render(camera_list[idx], gaussians, pipe, background)
                depth_3d = render_pkg["depth_3dgs"]
                rendered_depth_list.append(depth_3d.detach().squeeze().cpu().clone())

            # Filtering the edited images using ImageReward
            # get ranking and rewards
            with torch.cuda.amp.autocast(dtype=torch.float32):
                ranking, rewards = reward_model.inference_rank(ed.sampling_prompt, edited_image_pil_list_RM)

            # sampling images according to top-k ranking
            top_ratio = 1 - ed.filtering_ratio
            top_count = int(len(ranking) * top_ratio)
            is_top_selection = [rank <= top_count for rank in ranking]

        """Multi-View Fusion Guidance (MFG)"""
        edited_image_MFG_list = []
        for epoch in range(opt.epoch):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Multi-view reprojection progress")):

                gt_image = batch['gt_image'].to(self._execution_device)  # [1, 3, 512,512]
                idx = batch['idx'].item()  # current camera index
                viewpoint_cam = camera_list[idx]
                gt_image = F.interpolate(gt_image, size=(image_height, image_width), mode='bilinear', align_corners=True)

                # reprojecting
                with torch.cuda.amp.autocast(dtype=torch.float32):

                    src_cam_idx_list = []
                    dst_cam_idx = idx

                    while len(src_cam_idx_list) < 5:
                        for camera_idx in camera_dist_order[idx][1:]:
                            if is_top_selection[camera_idx]:
                                src_cam_idx_list.append(camera_idx)
                                if len(src_cam_idx_list) >= 5:
                                    break

                    dst_camera = camera_list[dst_cam_idx]
                    reprejected_pixels_list = []
                    reprejected_colors_list = []

                    for camera_idx in src_cam_idx_list:
                        camera = camera_list[camera_idx]

                        color = edited_image_list[camera_idx].detach()
                        depth = rendered_depth_list[camera_idx].squeeze()

                        reprejected_points, reprejected_colors = reproject_rgbd(
                            camera,
                            dst_camera,
                            color.to(self._execution_device),
                            depth.to(self._execution_device),
                        )

                        reprejected_pixels_list.append(reprejected_points)
                        reprejected_colors_list.append(reprejected_colors)

                    # reprojected image
                    dst_image, _ = reprojected2img(
                        reprejected_pixels_list,
                        reprejected_colors_list,
                        dst_camera,
                        alpha_blend=True,
                    )

                    dst_image_np = dst_image.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
                    dst_image_pil = Image.fromarray((dst_image_np * 255).astype(np.uint8))

                    reprejected_image = dst_image.unsqueeze(0)
                    mask, _, _, _ = lang_sam.predict(dst_image_pil, ed.target_mask_prompt)

                    try:
                        if mask.shape[0] != 1:
                            mask = mask[0]

                        if len(mask.shape) == 2:
                            mask = mask.unsqueeze(0)

                    except:
                        mask = torch.ones((1, image_height, image_width)).to(self._execution_device)

                    if ed.target_mask_prompt == "no_mask":
                        mask = torch.ones((1, image_height, image_width)).to(self._execution_device)

                    # background replacement
                    MF_image = reprejected_image * mask.to(reprejected_image.device)

                    mask_bool = mask.bool().to(self._execution_device)
                    MF_image = MF_image + (gt_image * ~mask_bool)  # (3, 512, 512)

                reset_attention_maps(self.unet)

                if MF_image.shape[2] != 512 or MF_image.shape[3] != 512:
                    MF_image = F.interpolate(MF_image, size=(512, 512), mode='bilinear', align_corners=True)

                if gt_image.shape[2] != 512 or gt_image.shape[3] != 512:
                    gt_image = F.interpolate(gt_image, size=(512, 512), mode='bilinear', align_corners=True)

                # MFG (Multi-View Fusion Guidance)
                edited_image_MFG = self.edit_image_MFG(  # edited_image_MFG -> torch.Size([1, 3, 512, 512])
                    trg_prompt_embeds,
                    gt_image,  # input
                    gt_image,  # image cond
                    MF_image,  # multi-view fused image cond
                    text_guidance_scale=ed.text_guidance_scale,
                    MFG_scale=ed.MFG_scale,
                    source_guidance_scale=ed.source_guidance_scale,
                    diffusion_steps=20,
                    lower_bound=0.70,
                    upper_bound=0.98,
                    step = step,
                    prompt = ed.target_prompt
                )

                edited_image_MFG = F.interpolate(edited_image_MFG.float(), size=(image_height, image_width), mode='bilinear',
                                                 align_corners=False).to(edited_image_MFG.dtype)

                # # save mfg edited images attention map
                trg_attention_map = get_all_attention_maps(self.unet)
                trg_attention_map_by_tokens = seperate_attention_maps_by_tokens(self.unet, trg_attention_map,
                                                                                self.tokenizer, ed.target_prompt)

                vis_path_trg = None

                vis_path_trg = "./output/"

                # get target object prompt attention map
                trg_object_average_attention_map, trg_object_average_attention_map_512 = save_attention_maps(
                    trg_attention_map_by_tokens, trg_attention_map, ed.object_prompt, output_dir=vis_path_trg,
                    image_height=image_height, image_width=image_width
                )

                trg_object_average_attention_map_512 = torch.tensor(trg_object_average_attention_map_512)

                # Min-Max Normalization: [0, 1]
                min_val = trg_object_average_attention_map_512.min()
                max_val = trg_object_average_attention_map_512.max()
                trg_object_average_attention_map_512 = (trg_object_average_attention_map_512 - min_val) / (
                        max_val - min_val)

                attn_list.append(trg_object_average_attention_map_512)

                if epoch == 0:
                    edited_image_MFG_list.append(edited_image_MFG.squeeze(0).detach().cpu().clone())

                gaussians.update_learning_rate(iteration)

                viewspace_point_list = []

                render_pkg = render(viewpoint_cam, gaussians, pipe, background)

                rendered_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                    "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                if rendered_image.shape[1] != image_height or rendered_image.shape[2] != image_width:
                    rendered_image = F.interpolate(rendered_image.to(torch.float32), size=(image_height, image_width),
                                                   mode='bilinear', align_corners=True)
                viewspace_point_list.append(viewspace_point_tensor)

                edited_image_MFG_for_3dgs = edited_image_MFG.squeeze(0).cuda().to(rendered_image.dtype)

                if edited_image_MFG_for_3dgs.shape[1] != image_height or edited_image_MFG_for_3dgs.shape[
                    2] != image_width:
                    edited_image_MFG_for_3dgs = F.interpolate(edited_image_MFG_for_3dgs.to(torch.float32),
                                                              size=(image_height, image_width), mode='bilinear',
                                                              align_corners=True)
                # calculate loss
                Ll1 = l1_loss(rendered_image, edited_image_MFG_for_3dgs)
                p_loss = lpips_loss_fn(torch.clamp(edited_image_MFG_for_3dgs, -1, 1),
                                       torch.clamp(rendered_image, -1, 1))

                total_loss = Ll1 + p_loss

                total_loss.backward()

                # optimization Step
                with torch.no_grad():
                    viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_list[0])
                    for idex in range(len(viewspace_point_list)):
                        viewspace_point_tensor_grad = (
                                viewspace_point_tensor_grad
                                + viewspace_point_list[idex].grad
                        )

                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gaussians.add_densification_stats(
                        viewspace_point_tensor_grad, visibility_filter
                    )

                    if iteration == start_iteration:
                        # Densification
                        gaussians.densify_and_prune(
                            0.001, 0.005, scene.cameras_extent, 5, is_first_densification=True, k_percent=opt.k_percent,
                            attn_thres=opt.attn_thres
                        )
                    elif iteration % opt.densification_interval == 0:
                        # Densification
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, 0.005, scene.cameras_extent, 5, is_first_densification=False,
                            k_percent=opt.k_percent, attn_thres=opt.attn_thres
                        )

                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                iteration = iteration + 1

            # clean GPU Resources
        del self.unet
        torch.cuda.empty_cache()

        '''Attention-Guided Trimming (AGT)'''
        # attention Weighting
        attn_weights = torch.zeros_like(gaussians._opacity)
        attn_weights_cnt = torch.zeros_like(gaussians._opacity, dtype=torch.int32)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Attention Weighting")):
            idx = batch['idx'].item()
            camera = camera_list[idx]

            attn_mask = attn_list[step].to(self._execution_device)
            temp_binary = attn_mask > 0.5
            attn_mask = attn_mask * temp_binary
            attn_mask = attn_mask.unsqueeze(0)

            gaussians.apply_weights(camera, attn_weights, attn_weights_cnt, attn_mask)

        attn_weights /= attn_weights_cnt + 1e-7
        selected_mask = attn_weights[:, 0]

        gaussians.set_mask(selected_mask)
        gaussians.apply_grad_mask(selected_mask)

        # save point_cloud
        print(f"\n[EPOCH {opt.epoch}] Saving Gaussians")
        scene.save(iteration)

        # save checkpoint
        print(f"\n[EPOCH {opt.epoch}] Saving Checkpoint\n")
        torch.save((gaussians.capture(), iteration),
                   scene.model_path + f"/point_cloud/iteration_{iteration}" + f"/chkpnt{iteration}.pth")

        # save rendering result
        render_sets(dataset, iteration, pipe, False, False, False)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = ArgumentParser(description="Editing Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ed = EditingParams(parser)
    args = parser.parse_args(sys.argv[1:])

    set_seed(0)

    pipeline = editsplat_Pipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.bfloat16).to(device)

    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
        shutil.copyfile(__file__, os.path.join(args.model_path, 'train_frozen.py'))
    import TDATTN_SGT
    import numpy as np
    from TDATTN_SGT import tdattn_sgt
    import csv

    tokenizer = pipeline.tokenizer
    reweight_steps = [0.0, 1.01]
    category_vectors = np.load("./HAB_scores/category_vectors_epoch_1_to_1_neg_prompt_False_merged_340_final_text_descriptions_sd_v1_4.npy")
    head_index = np.load("./HAB_scores/head_index_epoch_1_to_1_neg_prompt_False_merged_340_final_text_descriptions_sd_v1_4.npy")
    LOW_RESOURCE = False
    tdattn = tdattn_sgt(sh_degree=0)
    data = []
    description_file_path = "./HAB_scores/340_final_text_list.csv"
    with open(description_file_path, "r") as f:
        render1 = csv.reader(f)
        for row in render1:
            data.append(row)
    num_list = np.arange(1, 11).astype(str)
    data = [row for row in data if row != []]
    data = [row[0] for row in data if not any(num in row[0] for num in num_list)]
    data[0] = data[0].replace("\ufeff", "")
    SGT = data
    # model_version = "sd_v2_1_base"
    prompts = [args.target_prompt]
    selected_token = "face"
    # no_key = 3
    desired_concept = "Material"
    undesired_concept = "Material"
    adjusting_direction = [desired_concept,desired_concept, undesired_concept]
    equalizers = []
    zero_prompt_equalizer = TDATTN_SGT.get_equalizer(prompts[0], (selected_token), (0,), tokenizer=tokenizer).cuda()
    equalizers.append(zero_prompt_equalizer)
    equalizers.append(1 - zero_prompt_equalizer)
    _ = pipeline(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        ed=ed.extract(args),
    )

    print("\nEditing complete.")