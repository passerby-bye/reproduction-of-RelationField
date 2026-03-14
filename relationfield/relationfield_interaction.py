# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from GARField
#   (https://github.com/chungmin99/garfield
# Copyright (c) 2014 GARField authors, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

"""Helper functions for interacting/visualization with relationfield model."""
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import viser
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import scale_gradients_by_distance_squared
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.viewer.viewer_elements import *
from nerfstudio.cameras.rays import RaySamples
from relationfield.instance_field import GarFieldHeadNames
from relationfield.relationfield_model import RelationFieldModel
from relationfield.semantic_field import OpenNerfFieldHeadNames
from transformers import AutoModel


class RelationFieldClickScene(nn.Module):
    """UI for clicking on a scene (visualized as spheres).
    This needs to be a nn.Module to allow the viewer to register callbacks.
    """
    _click_handle: viser.GlbHandle
    _sg_click_handle: viser.GlbHandle
    _relation_click_handle: viser.GlbHandle
    _inst_handle: viser.GlbHandle
    _box_handle: viser.GlbHandle
    selected_location: np.ndarray
    click_emb: torch.Tensor
    scale_handle: ViewerSlider  # For getting the scale to query relationfield
    thresh_handle: ViewerSlider
    normalization_toggle: ViewerCheckbox
    instance_toggle: ViewerCheckbox
    clip_positives: ViewerText
    bert_positives: ViewerText
    model_handle: List[RelationFieldModel]  # Store as list to avoid circular children

    def __init__(
            self,
            device: torch.device,
            scale_handle: ViewerSlider,
            thresh_handle: ViewerSlider,
            model_handle: List[RelationFieldModel]
        ):
        super().__init__()
        self.add_click_button: ViewerButton = ViewerButton(
            name="Cluster Instance Click", cb_hook=self._add_click_cb, 
            visible=False
        )
        
        self.add_relation_click_button: ViewerButton = ViewerButton(
            name="Select relation reference", cb_hook=self._add_relation_click_cb
        )
        
        self.instance_toggle = ViewerCheckbox(
                name="Instance Toggle",
                default_value=False,
            )
        self.normalization_toggle = ViewerCheckbox(
                name="Normalize Activation",
                default_value=False,
        )
        
        self.similarity_dropdown = ViewerDropdown(name="Similarity Feature", default_value="openseg", options=["openseg", "clip", "instance"], cb_hook=self._dropdown_cb)
        self.viewer_control: ViewerControl = ViewerControl()
        
        self.scale_handle = scale_handle
        self.thresh_handle = thresh_handle
        self.model_handle = model_handle
        self.scale_handle.cb_hook = self._update_scale_vis

        self._click_handle = None
        self._sg_click_handle = None
        self._inst_handle = None
        self._box_handle = None
        self.selected_location = None
        self.selected_relation_samples = None
        self.click_emb = None
        self.click_pos = None
        self.show_instances = False
        self._dropdown_value = "openseg"
        self.device = device
        
        self.clip_positives = ViewerText("Object Query", "", cb_hook=self.text_cb)
        self.bert_positives = ViewerText("Relationship Query", "", cb_hook=self.text_cb_bert)
        
        self.positives = []
        self.relation_positives = []
        self.negatives =  ["object", "things", "stuff", "texture", "other","nothing", "empty", "texture", "photo", "image", "picture"]
        self.relation_negatives_general = ["none","next to","and"]
        self.relation_negatives = ["none","next to","and"]

        self.pos_embeds = None
        self.neg_embeds = None
        self.bert_pos_embds = None
        self.bert_neg_embds = None
        self.clip_model = None
        self.jina_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(torch.bfloat16)
        self.jina_encode =  lambda x: self.jina_model.encode(x, task='text-matching', truncate_dim=512)
        self.cluster_centers = None
        self.instances_labels = None
        self._cluster_instance_handles = []

        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Get the state_dict of the module, but exclude the inference model
        current_state_dict = super(RelationFieldClickScene, self).state_dict(destination, prefix, keep_vars)
        for key in list(current_state_dict.keys()):
            if 'clip_model' in key or 'clip_tokenizer' in key or 'jina_model' in key or 'jina_encode' in key:
                del current_state_dict[key]
        return current_state_dict

        
        
    def _dropdown_cb(self, dropdown: ViewerDropdown):
        self._dropdown_value = dropdown.value
        if self.click_pos:
            self._on_rayclick_sg(self.click_pos)
        self.clip_model = None
        
    
    def _add_click_cb(self, button: ViewerButton):
        """Button press registers a click event, which will add a sphere.
        Refer more to nerfstudio docs for more details. """
        self.add_click_button.set_disabled(True)
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)
            self.add_click_button.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)
        

    def _add_relation_click_cb(self, button: ViewerButton):
        """Button press registers a click event, which will add a sphere.
        Refer more to nerfstudio docs for more details. """
        self.add_relation_click_button.set_disabled(True)
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick_relation(click)
            self.add_relation_click_button.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)
        
    def _on_rayclick(self, click: ViewerClick):
        """On click, calculate the 3D position of the click and visualize it.
        Also keep track of the selected location."""

        origin = torch.tensor(click.origin).view(1, 3)
        direction = torch.tensor(click.direction).view(1, 3)

        # get intersection
        bundle = RayBundle(
            origin,
            direction,
            torch.tensor(0.001).view(1, 1),
            nears=torch.tensor(0.05).view(1, 1),
            fars=torch.tensor(100).view(1, 1),
            camera_indices=torch.tensor(0).view(1, 1),
        ).to(self.device)

        # Get the distance/depth to the intersection --> calculate 3D position of the click
        model = self.model_handle[0]
        ray_samples, _, _ = model.proposal_sampler(bundle, density_fns=model.density_fns)
        field_outputs = model.field.forward(ray_samples, compute_normals=model.config.predict_normals)
        if model.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        with torch.no_grad():
            depth = model.renderer_depth(weights=weights, ray_samples=ray_samples)
        distance = depth[0, 0].detach().cpu().numpy()
        click_position = np.array(origin + direction * distance) * VISER_NERFSTUDIO_SCALE_RATIO
        # Update click visualization
        self._del_click_cb(None)
        sphere_mesh: trimesh.Trimesh = trimesh.creation.icosphere(radius=0.1)

        sphere_mesh_handle = self.viewer_control.viser_server.add_mesh_simple(
                name="/hit_pos", 
                vertices=sphere_mesh.vertices,
                faces=sphere_mesh.faces,
                position=(np.array(origin + direction * distance) * VISER_NERFSTUDIO_SCALE_RATIO).flatten(),
                wireframe=True
            )
        self._click_handle = sphere_mesh_handle
        self.selected_location = np.array(origin + direction * distance)
        self._update_scale_vis(self.scale_handle)
        
    def _on_rayclick_relation(self, click: ViewerClick):
        """On click, calculate the 3D position of the click and visualize it.
        Also keep track of the selected location."""

        origin = torch.tensor(click.origin).view(1, 3)
        direction = torch.tensor(click.direction).view(1, 3)

        # get intersection
        bundle = RayBundle(
            origin,
            direction,
            torch.tensor(0.001).view(1, 1),
            nears=torch.tensor(0.05).view(1, 1),
            fars=torch.tensor(100).view(1, 1),
            camera_indices=torch.tensor(0).view(1, 1),
        ).to(self.device)

        # Get the distance/depth to the intersection --> calculate 3D position of the click
        model = self.model_handle[0]
        
        ray_samples, _, _ = model.proposal_sampler(bundle, density_fns=model.density_fns)
        field_outputs = model.field.forward(ray_samples, compute_normals=model.config.predict_normals)
        
        
        if model.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        def gather_fn_query(tens):
            return torch.gather(
                tens, -2, best_query_ids.expand(*best_query_ids.shape[:-1], tens.shape[-1])
            )
        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn_query, dataclass_fn)
        weights, best_query_ids = torch.topk(
            weights, 24, dim=-2, sorted=False
        )
        ray_samples: RaySamples = ray_samples._apply_fn_to_fields(
            gather_fn_query, dataclass_fn
        )
            
        with torch.no_grad():
            depth = model.renderer_depth(weights=weights, ray_samples=ray_samples)
        distance = depth[0, 0].detach().cpu().numpy()
        click_position = np.array(origin + direction * distance) * VISER_NERFSTUDIO_SCALE_RATIO
        self.selected_relation_samples = ray_samples
        self.selected_relation_position = np.array(origin + direction * distance)
        
        self._del_click_cb(None)
        sphere_mesh: trimesh.Trimesh = trimesh.creation.icosphere(radius=0.1)
        sphere_mesh_handle = self.viewer_control.viser_server.add_mesh_simple(
                name="/hit_pos", 
                vertices=sphere_mesh.vertices,
                faces=sphere_mesh.faces,
                position=(np.array(origin + direction * distance) * VISER_NERFSTUDIO_SCALE_RATIO).flatten(),
                wireframe=True
            )
        self._relation_click_handle = sphere_mesh_handle


    def _del_click_cb(self, button: ViewerButton):
        """Remove the click location and click visualizations."""
        if self._click_handle is not None:
            self._click_handle.remove()
        if self._sg_click_handle is not None:
            self._sg_click_handle.remove()
        self._click_handle = None
        if self._box_handle is not None:
            self._box_handle.remove()
        self._box_handle = None
        self.selected_location = None

    def _update_scale_vis(self, slider: ViewerSlider):
        """Update the scale visualization."""
        if self._box_handle is not None:
            self._box_handle.remove()
            self._box_handle = None
        if self.selected_location is not None:
            box_mesh = trimesh.creation.icosphere(radius=VISER_NERFSTUDIO_SCALE_RATIO*max(0.001, slider.value)/2, subdivision=0)
            self._box_handle = self.viewer_control.viser_server.add_mesh_simple(
                name="/hit_pos_box", 
                vertices=box_mesh.vertices,
                faces=box_mesh.faces,
                position=(self.selected_location * VISER_NERFSTUDIO_SCALE_RATIO).flatten(),
                wireframe=True
            )
    

    def get_outputs(self, outputs: dict, location: np.ndarray = None):
        """Visualize affinity between the selected 3D point and the points visibl in current rendered view."""

        if location is None:
            if self.selected_location is None:
                return None
            location = self.selected_location
  
        instance_scale = self.scale_handle.value
        
        # mimic the fields call
        grouping_field = self.model_handle[0].grouping_field
        positions = torch.tensor(location).view(1, 3).to(self.device)
        positions = grouping_field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        xs = [e(positions.view(-1, 3)) for e in grouping_field.enc_list]
        x = torch.concat(xs, dim=-1)
        x = x / x.norm(dim=-1, keepdim=True)
        instance_pass = grouping_field.get_mlp(x, torch.tensor([instance_scale]).to(self.device).view(1, 1))

        return {
            "instance_interact": torch.norm(outputs['instance'] - instance_pass.float(), p=2, dim=-1)
        }
        
        
    @torch.no_grad()
    def get_relation_outputs(self, outputs: dict, ray_samples: RaySamples, semantic_field_outputs: torch.Tensor):
        if self.selected_relation_samples is None or self.selected_relation_position is None:
            return None
        model = self.model_handle[0]

        
        if self.instance_toggle.value:
            instance_interact = self.get_outputs(outputs, location=self.selected_relation_position)["instance_interact"]
            mask = instance_interact < 0.5
            instance_query_samples = ray_samples[mask]
            if len(instance_query_samples) > 0:
                instance_query_samples = model.concatenate_ray_samples(self.selected_relation_samples, instance_query_samples)
                points = ray_samples.frustums.get_positions().detach().cpu().view(-1,3).numpy()
                query_positions = instance_query_samples.frustums.get_positions().detach().cpu().view(-1,3).numpy()
                if query_positions.shape[0] > 100:
                    batch_idx = np.random.choice(query_positions.shape[0], 100, replace=False)
                    query_positions = query_positions[batch_idx]
            else:
                points = ray_samples.frustums.get_positions().detach().cpu().view(-1,3).numpy()
                query_positions = self.selected_relation_samples.frustums.get_positions().detach().cpu().view(-1,3).numpy()
            rel_feats = []
            for i in range(0, points.shape[0], 1000):
                points_batch = points[i:i+1000]
                with torch.no_grad():
                    while True:
                        out = model.get_outputs_for_points_with_query_batch(points_batch, query_positions)
                        if out is not None:
                            rel_feats.append(out['relation'].cpu())
                            break
                    
            rel_feat = torch.cat(rel_feats, dim=0).to(self.device)
            rel_feat = rel_feat.view(*ray_samples.frustums.shape, -1)
            mask = mask.float()
        else:
            n_query_samples = model.config.num_feat_samples
            query_pos = torch.from_numpy(self.selected_relation_position[None]).cuda().repeat(1, n_query_samples, 1)
            positions = ray_samples.frustums.get_positions()
            # Chunk relation inference to reduce peak VRAM in interactive viewer mode.
            chunk_rays = 256
            rel_chunks = []
            for i in range(0, positions.shape[0], chunk_rays):
                rel_chunks.append(
                    model.relation_embedding_from_points(
                        positions[i:i + chunk_rays], query_pos, None
                    )
                )
            rel_feat = torch.cat(rel_chunks, dim=0)
            # rel_feat = model.relation_embedding(ray_samples, semantic_field_outputs, self.selected_relation_samples, None)
            mask = None
        return {
            "relation": rel_feat,
            "mask": mask
        }
        
        
    def get_outputs_similarity(self, ray_samples: RaySamples, outputs: dict):
        """Visualize affinity between the selected 3D point and the points visibl in current rendered view."""
        if self.click_emb is None:
            return None
        
        sim = F.normalize(outputs[self._dropdown_value],dim=-1)@F.normalize(self.click_emb,dim=-1).T
        return {
            "similarity": sim
        }
        
    def text_cb(self,element):
        self.set_positives(element.value.split(";"))
        
    def text_cb_bert(self,element):
        self.set_relation_positives(element.value.split(";"))
        
    def set_positives(self, text_list):
        self.positives = text_list

        if self.clip_model is None:
            print("Loading model")
            # self.clip_model, _ = clip.load("ViT-L/14@336px", device="cuda")
            if self._dropdown_value == "openseg":
                self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name='ViT-L-14-336', pretrained="openai", device="cuda")
                self.clip_tokenizer = open_clip.get_tokenizer(model_name='ViT-L-14-336')
            else:
                self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device="cuda")
                self.clip_tokenizer = open_clip.get_tokenizer(model_name='ViT-L-14')
                
        with torch.no_grad():
            tok_phrases = self.clip_tokenizer(self.positives).to("cuda")
            tok_negatives = self.clip_tokenizer(self.negatives).to("cuda")
            self.pos_embeds = self.clip_model.encode_text(tok_phrases)
            self.neg_embeds = self.clip_model.encode_text(tok_negatives)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)
        
    def set_relation_positives(self, text_list):
        self.relation_positives = [text_list[0]] # [text_list[0]]
        self.relation_negatives = self.relation_negatives_general + text_list[1:]
        if self.jina_model is None:
            from transformers import AutoModel
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                self.jina_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(torch.bfloat16)
        if len(self.relation_positives) > 0 and len(self.relation_negatives) > 0:
            with torch.no_grad():
                self.bert_pos_embds =  torch.from_numpy(np.stack([self.jina_encode(p) for p in self.relation_positives])).cuda()
                self.bert_neg_embds = torch.from_numpy(np.stack([self.jina_encode(p) for p in self.relation_negatives])).cuda() 
                self.bert_pos_embds /= self.bert_pos_embds.norm(dim=-1, keepdim=True)
                self.bert_neg_embds /= self.bert_neg_embds.norm(dim=-1, keepdim=True)        


    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        if self.pos_embeds is None or self.neg_embeds is None or self.clip_model is None:
            return None
        
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]
        
    def get_relevancy_bert(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        if len(self.relation_positives) == 0 or len(self.relation_negatives) == 0:
            return None
        if self.bert_pos_embds is None or self.bert_neg_embds is None:
            return None

        phrases_embeds = torch.cat([self.bert_pos_embds, self.bert_neg_embds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.relation_positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.relation_negatives))  # rays x N_phrase
        
        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.relation_negatives), 2))[
            :, 0, :
        ]
        
        
    def get_max_across(self, ray_samples, outputs):
        if len(self.positives) == 0:
            return 

        if self._dropdown_value not in ["openseg", "clip"]:
            return None
        openseg = outputs[self._dropdown_value]
        openseg = F.normalize(openseg, dim=-1)
        openseg_shape = openseg.dim()
        if openseg_shape == 3:
            rays, samples, emb_dim = openseg.shape
            openseg = openseg.view(-1, openseg.shape[-1])
            
        relevancy = self.get_relevancy(openseg, 0)
        if relevancy is not None:
            relevancy = relevancy[:,0]
            
        if openseg_shape == 3:
            relevancy = relevancy.view(rays, samples)
            relevancy = torch.sum(relevancy, dim=-1)

        if relevancy.dim() == 1:
            relevancy = relevancy.unsqueeze(-1)
    
        return {
            "relevancy": relevancy
        }
        
    def get_max_across_relation(self, ray_samples, outputs, weights=None):
        if len(self.relation_positives) == 0:
            return
        rel_feat = outputs["relation_map"]
        rel_feat = F.normalize(rel_feat, dim=-1)

        had_sample_dim = rel_feat.dim() == 3
        if had_sample_dim:
            rays, samples, _ = rel_feat.shape
            rel_feat_flat = rel_feat.view(-1, rel_feat.shape[-1])
        else:
            rel_feat_flat = rel_feat.view(-1, rel_feat.shape[-1])

        relevancy = self.get_relevancy_bert(rel_feat_flat, 0)
        if relevancy is not None:
            relevancy = relevancy[:, 0]
        else:
            return None

        render_weights = weights
        if render_weights is None:
            render_weights = outputs.get("weights")
        if render_weights is None:
            weights_list = outputs.get("weights_list")
            if isinstance(weights_list, list) and len(weights_list) > 0:
                render_weights = weights_list[-1]
        if render_weights is None:
            return None
        if render_weights.dim() == 3 and render_weights.shape[-1] == 1:
            render_weights = render_weights.squeeze(-1)

        if had_sample_dim:
            relevancy_samples = relevancy.view(rays, samples)
            norm_weights = render_weights / (render_weights.max(dim=1, keepdim=True)[0] + 1e-6)
            relevancy_raw = torch.sum(norm_weights * relevancy_samples, dim=-1)
        else:
            relevancy_raw = relevancy

        # scale relevancy by distance
        scale_by_dist = True
        if scale_by_dist:
            positions = ray_samples.frustums.get_positions().detach()
            positions = positions.view(-1, 3)
            location = torch.tensor(self.selected_relation_position).view(1, 3).to(self.device)
            distance = torch.norm(positions - location, dim=-1)
            
            # distance_scaled = 1/(1+distance)
            distance_scaler = torch.exp(-0.5*distance)
            n_rays, n_samples = ray_samples.frustums.shape[:2]
            distance_scaler = distance_scaler.view(n_rays, n_samples)
            if had_sample_dim:
                scaled_samples = relevancy_samples * distance_scaler
            else:
                scaled_samples = relevancy_raw.unsqueeze(1).repeat(1, n_samples) * distance_scaler
            relevancy_scaled = self.model_handle[0].renderer_mean(
                scaled_samples.unsqueeze(-1), render_weights.detach().unsqueeze(-1)
            ).squeeze(-1)

        if relevancy_raw.dim() == 1:
            relevancy_raw = relevancy_raw.unsqueeze(-1)
        if relevancy_scaled.dim() == 1:
            relevancy_scaled = relevancy_scaled.unsqueeze(-1)

        return {
            "relation_relevancy_raw": relevancy_raw,
            "relation_relevancy_scaled": relevancy_scaled,
        }
        
        
    @torch.no_grad()
    def query_position(self, position, model):
        query_pos = torch.from_numpy(position[None])/VISER_NERFSTUDIO_SCALE_RATIO
        query_pos_dist=model.semantic_field.spatial_distortion(query_pos)
        query_pos_norm = (query_pos_dist+2)/4
        
        xs = [e(query_pos_norm.view(-1, 3)) for e in model.semantic_field.clip_encs]
        x = torch.concat(xs, dim=-1)
        clip_pass = model.semantic_field.clip_net(x)
        clip_pass = clip_pass / torch.linalg.norm(clip_pass,dim=-1,keepdim=True)
        return clip_pass

    def overlay_activation_rgb(self, activation, rgb):
        # p = activation
        if activation is None:
            return rgb
        # normalize = False
        if self.normalization_toggle.value:
            # normalize the activation to the range 0-1, in one line
            activation = (activation - activation.min()) / (activation.max() - activation.min())
            
        p = torch.clip(activation - self.thresh_handle.value, 0, 1).squeeze()
        p = p / (p.max()+1e-6)
        p = torch.clip(p, 0, 0.85) # top 15 percentile is too dark
        overlay = torch.tensor(matplotlib.colormaps["turbo"].colors, device=p.device)[(p*255).to(torch.long)]
        mask = (p <= 0).squeeze()
        alpha = 0.35
        black_image = torch.zeros_like(overlay).to(overlay.device)
        overlay = 0.90 * overlay + 0.1 * rgb
        overlay[mask] = (1 - alpha) * rgb[mask] + alpha * black_image[mask]

        return torch.clip(overlay,0,1)
