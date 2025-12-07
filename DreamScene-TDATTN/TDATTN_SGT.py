import abc
import math
from typing import Optional, Union, Tuple, List, Dict
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
import cv2
import numpy as np
import torch
from PIL import Image
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from torch import nn
from utils.general_utils import build_rotation

LOW_RESOURCE = False
model_version = "sd_v2_1_base"
from scene_gaussian import GaussianModel

class tdattn_sgt(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        self.GSweights = None
        self.GSweights_cnt = None
        self.c_batchsize = None
        self.prompts = None
        # self.tokenizer = None
        self.viewpoint_cams = None
        self.iteration_cur = None
        self.azimuths = None
        self.polars = None
        self.smoothedGS = None
        self.stop_iter = 7000
        self.standard_sd = True

    # 3d-attn
    def clip_grad_norm(self, val):
        torch.nn.utils.clip_grad_norm_(self._xyz, val)
        torch.nn.utils.clip_grad_norm_(self._features_dc, val)
        torch.nn.utils.clip_grad_norm_(self._features_rest, val)
        torch.nn.utils.clip_grad_norm_(self._opacity, val)
        torch.nn.utils.clip_grad_norm_(self._scaling, val)
        torch.nn.utils.clip_grad_norm_(self._rotation, val)
    def camera2rasterizer(self, viewpoint_camera, bg_color: torch.Tensor, sh_degree: int = 0):
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        return rasterizer

    def text_under_image(self, image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
        h, w = image.shape
        offset = int(h * .2)
        img = np.ones((h + offset, w), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        img[:h] = image
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
        cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)

        return img

    def aggregate_attention(self, attention_store, res: int, from_where, is_cross: bool, select: int):
        out = []
        attention_maps = attention_store
        num_pixels = res ** 2
        len_p = self.c_batchsize
        stores = []
        for location in from_where:

            if self.c_batchsize == 1:
                for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                    if item.shape[1] == num_pixels:
                        if len_p == 1:
                            cross_maps = item.reshape(len_p, -1, res, res, item.shape[-1])[select]
                        out.append(cross_maps)
            else:
                outss = []
                for i in range(self.c_batchsize):
                    groups = []
                    for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                        groups.append(item[i])

                    group = torch.cat(groups, dim=0)
                    # cross_maps0 = group.reshape(-1, res, res, 77)
                    outs = group.sum(0) / group.shape[0]


                    outss.append(outs)

                outss_1 = torch.stack(outss, dim=0)
            # net_flag += 1
            stores.append(outss_1)
        if len_p == 1:
            out = torch.cat(out, dim=0)
            out = out.sum(0) / out.shape[0]
            return out.cpu()
        else:
            stores2 = (stores[0] + stores[1]) / 2
            return stores2.cpu()

    def show_cross_attention(self, tokenizer, save_path_iter_attn, attention_store, res: int, from_where,
                             select: int = 0, save_2d_attn=1, no_key: int = -1000):
        tokens = tokenizer.encode(self.prompts[select])
        decoder = tokenizer.decode
        attention_maps = self.aggregate_attention(attention_store, res, from_where, True, select)
        image_key = []
        images = []

        if self.c_batchsize == 1:
            if save_2d_attn:
                for i in range(len(tokens)):
                    image = attention_maps[:, :, i]
                    image = 255 * image / image.max()
                    image = image.unsqueeze(-1).expand(*image.shape, 3)
                    image = image.numpy().astype(np.uint8)
                    image = np.array(Image.fromarray(image).resize((256, 256)))
                    if i == no_key:
                        image_key = np.array(Image.fromarray(image).resize((512, 512)))
                    # view_images(image)
                    image = self.text_under_image(image, decoder(int(tokens[i])))
                    images.append(image)
                # image_tensor = [torch.from_numpy(image).permute(2, 0, 1).float() for image in images]
                concatenated_image = np.concatenate(images, axis=1)
                save_path_with_res = f"{save_path_iter_attn}_{res}x{res}.jpg"
                cv2.imwrite(save_path_with_res, concatenated_image)
            else:
                image = attention_maps[:, :, no_key]
                image = 255 * image / image.max()
                image = image.unsqueeze(-1).expand(*image.shape, 3)

                image = image.numpy().astype(np.uint8)

                image_key = np.array(Image.fromarray(image).resize((512, 512)))

        else:
            if save_2d_attn:
                B = attention_maps.shape[0]
                for b in range(B):
                    images = []
                    concatenated_image = None
                    for i in range(len(tokens)):
                        image = attention_maps[b, :, :, i]
                        image = 255 * image / image.max()

                        image = image.numpy().astype(np.uint8)

                        image = np.array(Image.fromarray(image).resize((256, 256)))

                        if i == no_key:
                            image_key_512 = np.array(Image.fromarray(image).resize((512, 512)))
                            image_key.append(image_key_512)

                        image = self.text_under_image(image, decoder(int(tokens[i])))
                        images.append(image)

                    concatenated_image = np.concatenate(images, axis=1)

                    save_path_with_res = f"{save_path_iter_attn}_{res}x{res}_img{b}.jpg"
                    cv2.imwrite(save_path_with_res, concatenated_image)


            else:
                B = attention_maps.shape[0]
                for b in range(B):
                    images = []
                    image = attention_maps[b, :, :, no_key]
                    image = 255 * image / image.max()
                    # image = image.unsqueeze(-1).expand(*image.shape, 3)
                    image = image.numpy().astype(np.uint8)
                    # image = np.stack([image] * 3, axis=-1)
                    image_key_512 = np.array(Image.fromarray(image).resize((512, 512)))
                    image_key.append(image_key_512)
        return image_key

    def register_attention_control(self, model, controller):

        # flaggg = self.flaggg
        def ca_forward(self, place_in_unet, count):
            to_out = self.to_out
            if type(to_out) is torch.nn.modules.container.ModuleList:
                to_out = to_out[0]
            else:
                to_out = self.to_out

            def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
                is_cross = encoder_hidden_states is not None
                layer = count  # layer index

                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(
                    encoder_hidden_states)  # (is_cross) encoder_hidden_states.shape = (2*batch_size, 77, 768)
                value = self.to_v(encoder_hidden_states)
                wo_head = key.shape[0]  # 2*batch_size

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)  # (is_cross) (2*batch_size*h, 77, 40)
                value = self.head_to_batch_dim(value)  # (is_cross) (2*batch_size*h, 77, 40)
                w_head = key.shape[0]  # 2*batch_size*h

                attention_probs = self.get_attention_scores(query, key, attention_mask)  # shape: [2*h, res*res, 77]

                attention_probs = controller(attention_probs, is_cross, place_in_unet, layer)  # Focus here!

                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = to_out(hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(1, 2).view(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states

            return forward



        class DummyController:

            def __call__(self, *args):
                return args[0]

            def __init__(self):
                self.num_att_layers = 0

        if controller is None:
            controller = DummyController()


        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'Attention':
                if hasattr(net_, 'to_k') and net_.to_k.in_features != net_.to_k.out_features:
                    net_.forward = ca_forward(net_, place_in_unet, count)
                    return count + 1
                else:
                    return count
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")

        controller.num_att_layers = cross_att_count

    def update_mask(self, camera, attn_2d) -> None:
        # if self.c_batchsize==1:
        self.apply_weights(camera, self.GSweights, self.GSweights_cnt, attn_2d)

    def apply_weights(self, camera, GSweights, GSweights_cnt, image_weights):
        rasterizer = self.camera2rasterizer(
            camera, torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        )
        rasterizer.apply_weights(
            self.get_xyz,
            None,
            self.get_opacity,
            None,
            GSweights,
            self.get_scaling,
            self.get_rotation,
            None,
            GSweights_cnt,
            image_weights,
        )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in ['background']:
                if group["name"] == name:
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if group["name"] not in ['background']:
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        self.GSweights = self.GSweights[mask]
        self.GSweights_cnt = self.GSweights_cnt[mask]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._target_features_dc = self._target_features_dc[valid_points_mask]
        self._target_features_rest = self._target_features_rest[valid_points_mask]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in ['background']:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                        dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def cat_tensors_to_GSweights(self, tensors_dict):
        optimizable_tensors = {}

        extension_tensor_cnt = tensors_dict["GSweights_cnt"]
        extension_tensor = tensors_dict["GSweights"]
        self.GSweights = torch.cat((self.GSweights, extension_tensor), dim=0)
        self.GSweights_cnt = torch.cat((self.GSweights_cnt, extension_tensor_cnt), dim=0)
        optimizable_tensors["GSweights"] = self.GSweights
        optimizable_tensors["GSweights_cnt"] = self.GSweights_cnt
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_t_features_dc=None, new_t_features_rest=None, new_GSweights=None,
                              new_GSweights_cnt=None):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "GSweights": new_GSweights,
             "new_GSweights_cnt": new_GSweights_cnt
             }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        gsweights_tensors = self.cat_tensors_to_GSweights(
            {"GSweights": new_GSweights, "GSweights_cnt": new_GSweights_cnt})
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.GSweights = gsweights_tensors["GSweights"]

        self.GSweights_cnt = gsweights_tensors["GSweights_cnt"]

        self._target_features_dc = torch.cat((self._target_features_dc, new_t_features_dc), dim=0)
        self._target_features_rest = torch.cat((self._target_features_rest, new_t_features_rest), dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_t_features_dc = self._target_features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_t_features_rest = self._target_features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_GSweights = self.GSweights[selected_pts_mask].repeat(N, 1)
        new_GSweights_cnt = self.GSweights_cnt[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_t_features_dc, new_t_features_rest, new_GSweights, new_GSweights_cnt)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_brightness(self, reset_rate):
        feature_new = self.get_features * reset_rate
        f_dc = self.replace_tensor_to_optimizer(feature_new[:, :1], "f_dc")
        f_rest = self.replace_tensor_to_optimizer(feature_new[:, 1:], "f_rest")
        self._features_dc = f_dc["f_dc"]
        self._features_rest = f_rest["f_rest"]
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_t_features_dc = self._target_features_dc[selected_pts_mask]
        new_t_features_rest = self._target_features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_GSweights = self.GSweights[selected_pts_mask]
        new_GSweights_cnt = self.GSweights_cnt[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_t_features_dc, new_t_features_rest, new_GSweights=new_GSweights,
                                   new_GSweights_cnt=new_GSweights_cnt)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str, layer):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, layer):
        if self.skip_call:
            return attn
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet, layer)
            else:
                h = attn.shape[0]
                attn[:h // 3] = self.forward(attn[:h // 3], is_cross, place_in_unet, layer, no_bc=0)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, skip_call):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.skip_call = skip_call


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, layer):
        key = f"{place_in_unet}_cross"
        if attn.shape[1] == 32 ** 2 and is_cross:  # avoid memory overhead
            attn_load1 = attn.reshape(4, 10, attn.shape[1], 77)
            attn_load2 = attn_load1.view(4, 10, 32, 32, 77)
            self.step_store[key].append(attn_load2)

        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):

        average_attention = {}

        for key in self.attention_store:
            attention_maps = self.attention_store[key]
            normalized_attention_maps = []
            for item in attention_maps:
                normalized_attention_maps.append(item / self.cur_step)

            average_attention[key] = normalized_attention_maps

        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, skip_call):
        super(AttentionStore, self).__init__(skip_call)
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        return x_t

    @abc.abstractmethod
    def replace_cross_attention(self, attn_replace, equlizer):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str, layer, no_bc):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet, layer)  ######先后顺序
        if is_cross:
            if self.standard_sd:
                return attn
            if model_version == "sd_v1_4" or model_version == "sd_v2_1_base":

                num_head = attn.shape[0] // self.batch_size  # attn.shape: (batch_size * num_head, res^2, 77)
                attn = attn.reshape(self.batch_size, num_head, *attn.shape[1:])
                alpha_words = self.cross_replace_alpha[self.cur_step]
                for head in range(num_head):
                    attn[:, head] = self.replace_cross_attention(attn[:, head], place_in_unet, layer,
                                                                 head) * alpha_words[:, 0] + (
                                            1 - alpha_words[:, 0]) * attn[:, head]
                print(f"batch:{no_bc}----place_in_unet:{place_in_unet}----layer:{layer}----{num_head}  OK!")
                attn = attn.reshape(self.batch_size * num_head, *attn.shape[2:])
                return attn
            else:
                raise ValueError("Not implemented yet")
        return attn

    def __init__(self, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]], skip_call,
                 batch_size, tokenizer):
        super(AttentionControlEdit, self).__init__(skip_call)
        self.batch_size = batch_size
        self.cross_replace_alpha = get_time_words_attention_alpha(num_steps, cross_replace_steps, tokenizer).cuda()


class AttentionReweight(AttentionControlEdit):
    """Reweight specific tokens in the attention map"""

    def replace_cross_attention(self, attn, place_in_unet=None, layer=None, head=None, SGT=None):
        head_pos = f"{place_in_unet}, layer: {layer}, head: {head}"
        desired_concept_rescaler = self.category_vectors[
            self.SGT.index(self.adjusting_direction[0]), np.where(self.head_index == head_pos)[0][0]]
        undesired_concept_rescaler = self.category_vectors[
            self.SGT.index(self.adjusting_direction[1]), np.where(self.head_index == head_pos)[0][0]]
        # rescale_factor = (self.equalizers[0][:, None, :] + self.equalizers[1][:, None, :] * (5 * desired_concept_rescaler))  # concept adjusting
        rescale_factor = (self.equalizers[0][:, None, :] +
                 self.equalizers[1][:, None, :] *
                 (0.7 if desired_concept_rescaler < 0.7 else
                  desired_concept_rescaler))

        attn = attn * rescale_factor

        return attn

    def __init__(self, num_steps: int, cross_replace_steps, equalizers, adjusting_direction=None,
                 category_vectors=None, head_index=None, standard_sd=None, skip_call=False, batch_size=1,
                 tokenizer=None, SGT=None):
        super(AttentionReweight, self).__init__(num_steps, cross_replace_steps, skip_call, batch_size, tokenizer)
        self.adjusting_direction = adjusting_direction
        self.equalizers = equalizers
        self.cross_replace_steps = cross_replace_steps
        self.category_vectors = category_vectors
        self.head_index = head_index
        self.standard_sd = standard_sd
        self.SGT = SGT
        # self.flagg = 0


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor] = None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    # for model v1.4, change [num_steps -> num_steps + 1]
    if model_version == "sd_v1_4" or model_version == "sd_v2_1_base":
        alpha_time_words = torch.zeros(num_steps + 1, 1, max_num_words)
    else:
        alpha_time_words = torch.zeros(num_steps, 1, max_num_words)
    for i in range(1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)

    if model_version == "sd_v1_4" or model_version == "sd_v2_1_base":
        alpha_time_words = alpha_time_words.reshape(num_steps + 1, 1, 1, 1, max_num_words)
    else:
        alpha_time_words = alpha_time_words.reshape(num_steps, 1, 1, 1, max_num_words)
    return alpha_time_words


def get_word_inds(text: str, word_place, tokenizer):
    """Return the index of 'word_place' in the tokenzied 'text'
       cf) 'word_place' may appear multiple times in the 'text'"""
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = word_place.strip()
        if len(word_place.split(" ")) == 1:
            word_place = [i for i, word in enumerate(split_text) if word_place == word]
        else:
            word_place_splited = word_place.split(" ")
            word_place_ = []
            for i, word in enumerate(split_text):
                if word == word_place_splited[0]:
                    if split_text[i:i + len(word_place_splited)] == word_place_splited:
                        word_place_ += [j for j in range(i, i + len(word_place_splited))]
            word_place = word_place_
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
Tuple[float, ...]], tokenizer):
    """"Equalizer for attention reweighting"""
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer
