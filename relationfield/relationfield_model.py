# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Literal, Union

from torch.nn import Parameter
import numpy as np
import torch
import torch.nn.functional as F
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.models.depth_nerfacto import DepthNerfactoModel, DepthNerfactoModelConfig

from nerfstudio.viewer.viewer_elements import *
from nerfstudio.model_components.losses import scale_gradients_by_distance_squared
from nerfstudio.field_components.activations import trunc_exp

from relationfield.instance_field import (
    GarField,
    GarFieldConfig,
)
from relationfield.relationfield_renderers import MeanRenderer, FeatureRenderer
from relationfield.semantic_field import OpenNerfField, OpenNerfFieldHeadNames
from relationfield.relation_field import RelationField
from relationfield.type_aliases import TensorType


if os.getenv("NERFACTO_DEPTH"):
    MODEL = DepthNerfactoModel
    CONFIG = DepthNerfactoModelConfig
else:
    MODEL = NerfactoModel
    CONFIG = NerfactoModelConfig



@dataclass
class RelationFieldModelConfig(CONFIG):
    _target: Type = field(default_factory=lambda: RelationFieldModel)
    instance_field: GarFieldConfig = GarFieldConfig()

    max_grouping_scale: float = 2.0
    """Maximum scale to use for grouping supervision. Should be set during pipeline init."""

    num_feat_samples: int = 24
    """Number of samples per ray to use for grouping supervision."""

    use_hierarchy_losses: bool = True
    use_single_scale: bool = False
    """For ablation only. For full relationfield, keep hierarchy=True and single_scale=False."""
    
    clip_loss_weight: float = 1.0
    openseg_loss_weight: float = 0.1 
    instance_loss_weight: float = 1.0
    relation_loss_weight: float = 1.0
    relation_lambda_negatives: float = 0.05
    dynamic_relation_lambda: bool = False
    relation_shared_enc: bool = False
    relation_semantic_feat: bool = False
    norm_feats: bool = False
    relation_occurance_weight: bool = False

    openseg_loss: Literal["Huber", "Cosine", "MSE"] = 'MSE'
    relation_loss: Literal["Huber", "Cosine", "MSE"] = 'MSE'
    if openseg_loss == 'Cosine':
        openseg_loss_weight*=0.01
    if relation_loss == 'Cosine':
        relation_loss_weight*=0.01
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_semantic_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)
    num_hidden_clip_layers: int = 1


class RelationFieldModel(MODEL):
    config: RelationFieldModelConfig
    grouping_field: GarField

    def populate_modules(self):
        super().populate_modules()
        self.renderer_feat = FeatureRenderer()
        self.renderer_mean = MeanRenderer()
        
        self.config.instance_field.use_single_scale = self.config.use_single_scale
        
        
        # Add a slider to the viewer to control the scale of the grouping field.
        self.scale_slider = ViewerSlider("Scale", 0.1, 0.0, 2.0, 0.001, visible=False)
        self.thresh_slider = ViewerSlider("Threshold", 0.5, 0.0, 1.0, 0.001)

        # Store reference to click interface for relationfield. 
        # Note the List[RelationFieldModel] is to avoid circular children.
        from relationfield.relationfield_interaction import RelationFieldClickScene
        self.click_scene: RelationFieldClickScene = RelationFieldClickScene(
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            scale_handle=self.scale_slider,
            thresh_handle=self.thresh_slider,
            model_handle=[self]
            )
        
        self.grouping_field = self.config.instance_field.setup()
        
        self.semantic_field = OpenNerfField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            self.config.num_hidden_clip_layers,
        )
        
        self.relation_field = RelationField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            3 if not self.config.relation_shared_enc else 8,
            self.config.relation_shared_enc,
            self.config.relation_semantic_feat,
        )
        
        
    def forward(self, ray_bundle: Union[RayBundle, Cameras], batch=None) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
            if batch is not None and 'query_bundle' in batch:
                batch['query_bundle'] = self.collider(batch['query_bundle'])

        return self.get_outputs(ray_bundle, batch)

    @torch.autocast('cuda')
    def get_outputs(self, ray_bundle: RayBundle, batch=None) -> Dict[str, TensorType]:
        if batch is not None and 'query_bundle' in batch:
            # concatentate the query bundle with the ray bundle
            new_metadata = ray_bundle.metadata.copy()
            new_metadata['n_query_rays'] = batch['query_bundle'].metadata['n_query_rays']
            new_metadata['directions_norm'] = torch.cat([ray_bundle.metadata['directions_norm'], torch.ones((new_metadata['n_query_rays'],1)).to(self.device)], dim=0)
            new_metadata['scale'] = torch.cat([ray_bundle.metadata['scale'], torch.ones(new_metadata['n_query_rays']).to(self.device)], dim=0)
            ray_bundle = RayBundle(origins=torch.cat([ray_bundle.origins, batch['query_bundle'].origins], dim=0), 
                                   directions=torch.cat([ray_bundle.directions, batch['query_bundle'].directions], dim=0),
                                   pixel_area=torch.cat([ray_bundle.pixel_area, batch['query_bundle'].pixel_area], dim=0),
                                   nears=torch.cat([ray_bundle.nears, batch['query_bundle'].nears], dim=0),
                                   fars=torch.cat([ray_bundle.fars, batch['query_bundle'].fars], dim=0),
                                   camera_indices=torch.cat([ray_bundle.camera_indices, batch['query_bundle'].camera_indices], dim=0),
                                   metadata=new_metadata
                                )
            
        outputs = super().get_outputs(ray_bundle)

        if self.grouping_field.quantile_transformer is None:
            # If scale statistics are not available, it's not possible to calculate grouping features.
            return outputs

        # Recalculate ray samples and weights
        # ... only if the model is in eval mode, where it should be no_grad(). 
        # If in training mode, `outputs` should already have calculated ray samples and weights.
        # Without this if-block, camera optimizer? gradients? seem to get messed up.
        ray_samples: RaySamples
        if self.training:
            ray_samples, weights = outputs["ray_samples_list"][-1], outputs["weights_list"][-1]
        else:
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
            if self.config.use_gradient_scaling:
                field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            
        if batch is not None and 'query_bundle' in batch:
            n_ray_bundle_rays = ray_samples.shape[0] - batch['query_bundle'].shape[0]
            ray_samples, query_ray_samples = ray_samples[:n_ray_bundle_rays], ray_samples[n_ray_bundle_rays:]
            weights, query_weights = weights[:n_ray_bundle_rays], weights[n_ray_bundle_rays:]
            
            # outputs_query = {}
            for key in outputs.keys():
                if type(outputs[key]) is list:
                    # outputs_query[key] = [outputs[key][i][n_ray_bundle_rays:] for i in range(len(outputs[key]))]
                    outputs[key] = [outputs[key][i][:n_ray_bundle_rays] for i in range(len(outputs[key]))]
                    continue
                # outputs_query[key] = outputs[key][n_ray_bundle_rays:]
                outputs[key] = outputs[key][:n_ray_bundle_rays]

        # Choose the top k samples with the highest weights, to be used for grouping.
        # This is to decrease # of samples queried for grouping, while sampling close to the scene density.
        def gather_fn(tens):
            return torch.gather(
                tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1])
            )

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        field_weights, best_ids = torch.topk(
            weights, self.config.num_feat_samples, dim=-2, sorted=False
        )
        field_samples: RaySamples = ray_samples._apply_fn_to_fields(
            gather_fn, dataclass_fn
        )
        if batch is not None and 'query_bundle' in batch:
            def gather_fn_query(tens):
                return torch.gather(
                    tens, -2, best_query_ids.expand(*best_query_ids.shape[:-1], tens.shape[-1])
                )
            dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn_query, dataclass_fn)
            query_top_weights, best_query_ids = torch.topk(
                query_weights, self.config.num_feat_samples, dim=-2, sorted=False
            )
            query_samples: RaySamples = query_ray_samples._apply_fn_to_fields(
                gather_fn_query, dataclass_fn
            )
            

        # Define the scale for each sample. If the scale is not provided, use the selected scale.
        # "scale" is included in ray_bundle.metadata only from training batches, but
        # this would be good way to override the scale during inference.
        if self.training and ("scale" in ray_bundle.metadata):
            scales = ray_bundle.metadata["scale"]
            if batch is not None and 'query_bundle' in batch:
                scales = scales[0,:n_ray_bundle_rays]
            instance_scales = scales.view(field_samples.shape[0], 1)
        elif "scale" in ray_bundle.metadata:
            if batch is not None and 'query_bundle' in batch:
                scales = scales[0,:n_ray_bundle_rays]
            scales = ray_bundle.metadata["scale"]
            instance_scales = scales.view(field_samples.shape[0], 1)
        else:
            slider_value = self.scale_slider.value
            instance_scales = (
                torch.ones(field_samples.shape[0], 1, device=self.device)
                * slider_value
            )

        # Calculate features for the scale-conditioned grouping field.
        # Hash values need to be included in the outputs for the loss calculation.
        hash = self.grouping_field.get_hash(field_samples)
        hash_rendered = self.renderer_feat(
            embeds=hash, weights=field_weights.detach().half()
        )
        
        if self.training:
            outputs["instance_hash"] = hash_rendered  # normalized!
            
        outputs["instance"] = self.grouping_field.get_mlp(hash_rendered, instance_scales).float()
            
        semantic_field_outputs = self.semantic_field.get_outputs(field_samples)
        outputs["openseg"] = self.renderer_mean(
            embeds=semantic_field_outputs[OpenNerfFieldHeadNames.OPENSEG], weights=field_weights.detach()
        )        
        if self.training and batch is not None and 'query_bundle' in batch:
            if not self.config.relation_shared_enc:
                relation_feature = self.relation_embedding(field_samples, semantic_field_outputs[OpenNerfFieldHeadNames.OPENSEG], query_samples, mask=None)
            else:
                relation_feature = self.relation_embedding_shared_enc(field_samples, semantic_field_outputs[OpenNerfFieldHeadNames.OPENSEG], query_samples, mask=None)

            outputs["relation"] = self.renderer_mean(embeds=relation_feature, weights=field_weights.detach())
            
        with torch.no_grad():
            # Interactive scene clicking
            click_output = self.click_scene.get_outputs(outputs)
            if click_output is not None:
                outputs.update(click_output)
                
            relation_click_output = self.click_scene.get_relation_outputs(outputs, field_samples,semantic_field_outputs[OpenNerfFieldHeadNames.OPENSEG])
            if relation_click_output is not None:
                relation_click_output['relation_map'] = self.renderer_mean(embeds=relation_click_output['relation'], weights=field_weights.detach()).float()
                del relation_click_output['relation']
                outputs.update(relation_click_output)
                
            relavancy_rel_outputs = self.click_scene.get_max_across_relation(field_samples,outputs,field_weights)
            if relavancy_rel_outputs is not None:
                outputs.update(relavancy_rel_outputs)
                outputs["rgb_rel_relevancy_raw"] = self.click_scene.overlay_activation_rgb(outputs["relation_relevancy_raw"],outputs["rgb"])
                outputs["rgb_rel_relevancy_scaled"] = self.click_scene.overlay_activation_rgb(outputs["relation_relevancy_scaled"],outputs["rgb"])
                
            relavancy_outputs = self.click_scene.get_max_across(ray_samples,outputs)
            if relavancy_outputs is not None:
                outputs.update(relavancy_outputs)
                outputs["rgb_relevancy"] = self.click_scene.overlay_activation_rgb(outputs["relevancy"],outputs["rgb"])
                
            emb_click_output = self.click_scene.get_outputs_similarity(field_samples,outputs)  
            if emb_click_output is not None:
                outputs.update(emb_click_output)
        return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples):
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        FieldHeadNames.UNCERTAINTY

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        return field_outputs, outputs, weights
    
    @torch.no_grad()
    def get_grouping_at_points(self, positions: TensorType, scale: float) -> TensorType:
        """Get the grouping features at a set of points, given a scale."""
        # Apply distortion, calculate hash values, then normalize
        positions = self.grouping_field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        xs = [e(positions.view(-1, 3)) for e in self.grouping_field.enc_list]
        x = torch.concat(xs, dim=-1)
        x = x / x.norm(dim=-1, keepdim=True)

        # Calculate grouping features; create a scale tensor to match the batch size
        instance_scale = torch.ones((x.shape[0], 1), device=self.device) * scale
        return self.grouping_field.get_mlp(x, instance_scale)
    
    
    def relation_embedding(self, ray_samples: RaySamples, semantic_embeddings: torch.Tensor, query_samples: RaySamples, mask: torch.Tensor=None) -> torch.Tensor:
        """Calculate the relation embedding between semantic embeddings and query positions."""
        # get semantic embeddings at query positions
        if mask is None:
            mask = torch.ones(semantic_embeddings.shape[0], dtype=torch.bool, device=self.device)
            
        query_pos = query_samples.frustums.get_positions().detach()

        query_pos = self.relation_field.spatial_distortion(query_pos)
        query_pos = (query_pos + 2.0) / 4.0
        
        positions = ray_samples.frustums.get_positions().detach()[mask]
        positions = self.relation_field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        query_pos = self._align_query_positions(query_pos, positions)

        xs = torch.concat([e(query_pos.reshape(-1, 3)) for e in self.semantic_field.clip_encs],dim=-1)
        query_embd = self.semantic_field.openseg_net(xs)

        query_positions = torch.cat([e(query_pos.reshape(-1, 3)) for e in self.relation_field.encs],dim=-1)
        
        semantic_embd = semantic_embeddings[mask].view(-1,semantic_embeddings.shape[-1])
        field_positions = torch.concat([e(positions.reshape(-1, 3)) for e in self.relation_field.encs], dim=-1) 
        
        if self.config.relation_semantic_feat:
            relation_pre_embd = torch.cat((
                query_embd, # maybe not needed, cut for efficiency
                query_positions,
                semantic_embd, # maybe not needed, cut for efficiency
                field_positions,
            ), dim=-1)
        else:
            relation_pre_embd = torch.cat((
                query_positions,
                field_positions,
            ), dim=-1)
        rel_feat = self.relation_field.relation_net(relation_pre_embd)
        return rel_feat.view(*ray_samples[mask].shape, -1)

    def _align_query_positions(self, query_pos: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Broadcast/repeat query positions to match positions shape [n_rays, n_samples, 3]."""
        if query_pos.dim() == 2:
            query_pos = query_pos.unsqueeze(1)
        if positions.dim() == 2:
            positions = positions.unsqueeze(1)

        target_rays, target_samples = positions.shape[0], positions.shape[1]
        q_rays, q_samples = query_pos.shape[0], query_pos.shape[1]

        if q_rays != target_rays:
            if q_rays == 1:
                query_pos = query_pos.expand(target_rays, -1, -1)
            else:
                query_pos = query_pos[:target_rays]

        if q_samples != target_samples:
            if q_samples == 1:
                query_pos = query_pos.expand(-1, target_samples, -1)
            else:
                repeats = (target_samples + q_samples - 1) // q_samples
                query_pos = query_pos.repeat(1, repeats, 1)[:, :target_samples, :]

        return query_pos

    def relation_embedding_from_points(self, ray_samples: torch.tensor, query_samples: torch.tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """Calculate the relation embedding between semantic embeddings and query positions."""
        # get semantic embeddings at query positions
        if mask is None:
            mask = torch.ones(ray_samples.shape[0], dtype=torch.bool, device=self.device)
            
        query_pos = query_samples.detach()

        query_pos = self.relation_field.spatial_distortion(query_pos)
        query_pos = (query_pos + 2.0) / 4.0
        
        positions = ray_samples.detach()[mask]
        positions = self.relation_field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        query_pos = self._align_query_positions(query_pos, positions)


        query_positions = torch.cat([e(query_pos.reshape(-1, 3)) for e in self.relation_field.encs],dim=-1)
        
        field_positions = torch.concat([e(positions.reshape(-1, 3)) for e in self.relation_field.encs], dim=-1) 
        
        
        relation_pre_embd = torch.cat((
            query_positions,
            field_positions,
        ), dim=-1)
        rel_feat = self.relation_field.relation_net(relation_pre_embd)
        
        return  rel_feat.view(*ray_samples[mask].shape[:-1], -1)


    def relation_embedding_shared_enc(self, ray_samples: RaySamples, semantic_embeddings: torch.Tensor, query_samples: RaySamples, mask: torch.Tensor=None) -> torch.Tensor:
        """Calculate the relation embedding between semantic embeddings and query positions."""
        # get semantic embeddings at query positions
        if mask is None:
            mask = torch.ones(semantic_embeddings.shape[0], dtype=torch.bool, device=self.device)
            
        query_pos = query_samples.frustums.get_positions().detach()
        query_pos = self.relation_field.spatial_distortion(query_pos)
        query_pos = (query_pos + 2.0) / 4.0
        
        positions = ray_samples.frustums.get_positions().detach()[mask]
        positions = self.relation_field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        query_pos = self._align_query_positions(query_pos, positions)
        xs = torch.cat(
            [e(torch.concat((positions.reshape(-1, 3), query_pos.reshape(-1, 3)), dim=-1)) for e in self.relation_field.encs],
            dim=-1,
        )

        rel_feat = self.relation_field.relation_net(xs)
        return rel_feat.view(*ray_samples[mask].shape, -1)
            
        

    def get_loss_dict_group(self, outputs, batch, metrics_dict=None):
        # loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if not self.training:
            return

        loss_dict = {}
        margin = 1.0

        ####################################################################################
        # Calculate GT labels for the positive and negative pairs
        ####################################################################################
        # TODO(cmk) want to make this a little more efficient and cleaner
        input_id1 = input_id2 = batch["mask_id"]

        # Expand labels
        labels1_expanded = input_id1.unsqueeze(1).expand(-1, input_id1.shape[0])
        labels2_expanded = input_id2.unsqueeze(0).expand(input_id2.shape[0], -1)

        # Mask for positive/negative pairs across the entire matrix
        mask_full_positive = labels1_expanded == labels2_expanded
        mask_full_negative = ~mask_full_positive

        # Create a block mask to only consider pairs within the same image -- no cross-image pairs
        chunk_size = batch["nPxImg"]  # i.e., the number of rays per image
        num_chunks = input_id1.shape[0] // chunk_size  # i.e., # of images in the batch
        block_mask = torch.kron(
            torch.eye(num_chunks, device=self.device, dtype=bool),
            torch.ones((chunk_size, chunk_size), device=self.device, dtype=bool),
        )  # block-diagonal matrix, to consider only pairs within the same image
        
        # Only consider upper triangle to avoid double-counting
        block_mask = torch.triu(block_mask, diagonal=0)  
        # Only consider pairs where both points are valid (-1 means not in mask / invalid)
        block_mask = block_mask * (labels1_expanded != -1) * (labels2_expanded != -1)

        # Mask for diagonal elements (i.e., pairs of the same point).
        # Don't consider these pairs for grouping supervision (pulling), since they are trivially similar.
        diag_mask = torch.eye(block_mask.shape[0], device=self.device, dtype=bool)

        hash_rendered = outputs["instance_hash"]
        scale = batch["scale"].view(-1, 1)

        ####################################################################################
        # Grouping supervision
        ####################################################################################
        total_loss = 0

        # 1. If (A, s_A) and (A', s_A) in same group, then supervise the features to be similar
        # Note that `use_single_scale` (for ablation only) causes grouping_field to ignore the scale input.
        instance = self.grouping_field.get_mlp(hash_rendered, scale)
        mask = torch.where(mask_full_positive * block_mask * (~diag_mask))
        instance_loss_1 = torch.norm(
            instance[mask[0]] - instance[mask[1]], p=2, dim=-1
        ).nansum()
        total_loss += instance_loss_1

        # 2. If ", then also supervise them to be similar at s > s_A
        if self.config.use_hierarchy_losses and (not self.config.use_single_scale):
            scale_diff = torch.max(
                torch.zeros_like(scale), (self.config.max_grouping_scale - scale)
            )
            larger_scale = scale + scale_diff * torch.rand(
                size=(1,), device=scale.device
            )
            instance = self.grouping_field.get_mlp(hash_rendered, larger_scale)
            mask = torch.where(mask_full_positive * block_mask * (~diag_mask))
            instance_loss_2 = torch.norm(
                instance[mask[0]] - instance[mask[1]], p=2, dim=-1
            ).nansum()
            total_loss += instance_loss_2

        # 4. Also supervising A, B to be dissimilar at scales s_A, s_B respectively seems to help.
        instance = self.grouping_field.get_mlp(hash_rendered, scale)
        mask = torch.where(mask_full_negative * block_mask)
        instance_loss_4 = (
            F.relu(
                margin - torch.norm(instance[mask[0]] - instance[mask[1]], p=2, dim=-1)
            )
        ).nansum()
        total_loss += instance_loss_4

        loss_dict["instance_loss"] = self.config.instance_loss_weight *(total_loss / torch.sum(block_mask).float())

        if self.config.instance_loss_weight == 0.0:
            del loss_dict["instance_loss"]

        return loss_dict
    
    def get_loss_dict_segmentation(self, outputs, batch, metrics_dict=None):
        if self.config.norm_feats:
            outputs["openseg"] = F.normalize(outputs["openseg"], dim=-1)
            outputs["clip"] = F.normalize(outputs["clip"], dim=-1)
        loss_dict = {}
        if self.training:
            
            if self.config.openseg_loss == 'Huber':
                unreduced_openseg = self.config.openseg_loss_weight * torch.nn.functional.huber_loss(
                    outputs["openseg"], batch["openseg"], delta=1.25, reduction="none")
            elif self.config.openseg_loss == 'Cosine':
                unreduced_openseg = self.config.openseg_loss_weight * (1.0 - torch.nn.functional.cosine_similarity(
                    outputs["openseg"], batch["openseg"]))                
            elif self.config.openseg_loss == 'MSE':
                unreduced_openseg = self.config.openseg_loss_weight * torch.nn.functional.mse_loss(
                    outputs["openseg"], batch["openseg"], reduction="none")

            #manually clip gradients
            unreduced_openseg = torch.clamp(unreduced_openseg, -10.0, 10.0)
            loss_dict["openseg_loss"] = unreduced_openseg.nansum(dim=-1).nanmean()
            
            if self.config.openseg_loss_weight == 0.0:
                del loss_dict["openseg_loss"]
            
        return loss_dict
    
    
    def get_loss_dict_relation(self, outputs, batch, metrics_dict=None):
        if self.config.norm_feats:
            outputs["relation"] = F.normalize(outputs["relation"], dim=-1)

        mask = batch["query_mask"]
        loss_dict = {}
        if self.training:
             
            if self.config.relation_loss == 'Huber':
                unreduced_relation = self.config.relation_loss_weight * torch.nn.functional.huber_loss(
                    outputs["relation"], batch["relation_embd"], delta=1.25, reduction="none")
            elif self.config.relation_loss == 'Cosine':
                unreduced_relation = self.config.relation_loss_weight * (1.0 - torch.nn.functional.cosine_similarity(
                    outputs["relation"], batch["relation_embd"]))                
            elif self.config.relation_loss == 'MSE':
                unreduced_relation = self.config.relation_loss_weight * torch.nn.functional.mse_loss(
                    outputs["relation"], batch["relation_embd"], reduction="none")
               
            #manually clip gradiants
            unreduced_relation = torch.clamp(unreduced_relation, -10.0, 10.0)
            
            
            # balance none relationships with semantic relationships
            relation_lambda = mask.sum() / (~mask).sum() if self.config.dynamic_relation_lambda else self.config.relation_lambda_negatives
            unreduced_relation[~mask] = relation_lambda*unreduced_relation[~mask]
            
            if self.config.relation_occurance_weight:
                unreduced_relation = batch["rel_weight"].unsqueeze(1) * unreduced_relation
            
            loss_dict["relation_loss"] = unreduced_relation.nansum(dim=-1).nanmean()

            if self.config.relation_loss_weight == 0.0:
                del loss_dict["relation_loss"]

        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["relationfield"] = list(self.grouping_field.parameters()) + list(self.semantic_field.parameters()) + list(self.relation_field.parameters())
        return param_groups

    def concatenate_ray_samples(self, ray_samples1: RaySamples, ray_samples2: RaySamples) -> RaySamples:
        # Concatenate Frustums
        frustums1, frustums2 = ray_samples1.frustums, ray_samples2.frustums
        concatenated_frustums = Frustums(
            origins=torch.cat([frustums1.origins, frustums2.origins], dim=0),
            directions=torch.cat([frustums1.directions, frustums2.directions], dim=0),
            starts=torch.cat([frustums1.starts, frustums2.starts], dim=0),
            ends=torch.cat([frustums1.ends, frustums2.ends], dim=0),
            pixel_area=torch.cat([frustums1.pixel_area, frustums2.pixel_area], dim=0),
            offsets=torch.cat([frustums1.offsets, frustums2.offsets], dim=0) if frustums1.offsets is not None and frustums2.offsets is not None else None
        )

        # Concatenate other optional tensors
        camera_indices = torch.cat([ray_samples1.camera_indices, ray_samples2.camera_indices], dim=0) if ray_samples1.camera_indices is not None and ray_samples2.camera_indices is not None else None
        deltas = torch.cat([ray_samples1.deltas, ray_samples2.deltas], dim=0) if ray_samples1.deltas is not None and ray_samples2.deltas is not None else None
        spacing_starts = torch.cat([ray_samples1.spacing_starts, ray_samples2.spacing_starts], dim=0) if ray_samples1.spacing_starts is not None and ray_samples2.spacing_starts is not None else None
        spacing_ends = torch.cat([ray_samples1.spacing_ends, ray_samples2.spacing_ends], dim=0) if ray_samples1.spacing_ends is not None and ray_samples2.spacing_ends is not None else None
        metadata = {key: torch.cat([ray_samples1.metadata[key], ray_samples2.metadata[key]], dim=0) if type(ray_samples1.metadata[key])==torch.Tensor else ray_samples1.metadata[key] for key in ray_samples1.metadata}
        times = torch.cat([ray_samples1.times, ray_samples2.times], dim=0) if ray_samples1.times is not None and ray_samples2.times is not None else None

        # Create a new RaySamples object
        concatenated_ray_samples = RaySamples(
            frustums=concatenated_frustums,
            camera_indices=camera_indices,
            deltas=deltas,
            spacing_starts=spacing_starts,
            spacing_ends=spacing_ends,
            spacing_to_euclidean_fn=ray_samples1.spacing_to_euclidean_fn,  # assuming these are the same for both
            metadata=metadata,
            times=times
        )

        return concatenated_ray_samples

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.
        This is the same as the base model's, but with a try/except in the case the shape is incorrect.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        min_chunk_size = 256
        while True:
            outputs_lists = defaultdict(list)
            try:
                for i in range(0, num_rays, num_rays_per_chunk):
                    start_idx = i
                    end_idx = i + num_rays_per_chunk
                    ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                    # move the chunk inputs to the model device
                    ray_bundle = ray_bundle.to(self.device)
                    outputs = self.forward(ray_bundle=ray_bundle, batch=None)
                    for output_name, output in outputs.items():  # type: ignore
                        if not isinstance(output, torch.Tensor):
                            # TODO: handle lists of tensors as well
                            continue
                        # move the chunk outputs from the model device back to the device of the inputs.
                        outputs_lists[output_name].append(output.to(input_device))
                break
            except RuntimeError as err:
                is_oom = "out of memory" in str(err).lower()
                if (not is_oom) or num_rays_per_chunk <= min_chunk_size:
                    raise
                num_rays_per_chunk = max(min_chunk_size, num_rays_per_chunk // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            cat_output = torch.cat(outputs_list, dim=0)
            # Keep outputs even when viewer ray bundles are flattened/cropped and
            # cannot be reshaped back to (H, W, ...). This prevents missing keys
            # such as "depth" that viewer interaction depends on.
            if cat_output.shape[0] == image_height * image_width:
                outputs[output_name] = cat_output.view(image_height, image_width, *cat_output.shape[1:])
            else:
                outputs[output_name] = cat_output
        return outputs
    
    
    @torch.no_grad()
    def get_outputs_for_points(self, points_batch, scale=0.5) -> Dict[str, torch.Tensor]:
        
        query_pos = torch.from_numpy(points_batch).cuda()
        query_pos_dist= self.semantic_field.spatial_distortion(query_pos)
        query_pos_norm = (query_pos_dist+2.)/4.
        
        
        h = self.field.mlp_base(query_pos_norm.view(-1, 3))
        density_before_activation, base_mlp_out = torch.split(h, [1, self.field.geo_feat_dim], dim=-1)
        density = self.field.average_init_density * trunc_exp(density_before_activation.to('cuda'))
        
        xs = [e(query_pos_norm.view(-1, 3)) for e in self.semantic_field.clip_encs]
        x = torch.concat(xs, dim=-1)
        
        clip_pass = self.semantic_field.clip_net(x)
        clip_pass = clip_pass / torch.linalg.norm(clip_pass,dim=-1,keepdim=True)
        
        openseg_pass = self.semantic_field.openseg_net(x)
        openseg_pass = openseg_pass / torch.linalg.norm(openseg_pass,dim=-1,keepdim=True)
        
        xs = [e(query_pos_norm.view(-1, 3)) for e in self.grouping_field.enc_list]
        x = torch.concat(xs, dim=-1)
        
        x = x / x.norm(dim=-1, keepdim=True)
        instance_scale = torch.ones((x.shape[0], 1), device=x.device) * scale
        instance_pass = self.grouping_field.get_mlp(x, instance_scale.to(x.device))

        return {
                "clip": clip_pass.view(*points_batch.shape[:2], -1),
                "openseg": openseg_pass.view(*points_batch.shape[:2], -1),
                "instance": instance_pass.view(*points_batch.shape[:2], -1),
                "density": density.view(*points_batch.shape[:2], -1)
        }

    @torch.no_grad()
    def get_outputs_for_points_with_query(self, points_batch, query, scale=0.5) -> Dict[str, torch.Tensor]:
        """
        get point features for a given query point
        """
        # add spatial distortion
        position_pos = torch.from_numpy(points_batch).cuda()
        position_pos_dist = self.semantic_field.spatial_distortion(position_pos)
        position_pos_norm = (position_pos_dist+2.)/4.
        # add spatial distortion to query point
        query_pos = torch.from_numpy(query).cuda()
        query_pos = query.unsqueeze(0).repeat(query_pos.shape[0],1)
        query_pos_dist = self.relation_field.spatial_distortion(query_pos)
        query_pos_norm = (query_pos_dist+2.)/4.
        
        semantic_dict = self.get_outputs_for_points(position_pos, None, None, scale)

        # compute relation features
        field_positions = torch.concat([e(position_pos_norm.view(-1, 3)) for e in self.relation_field.encs], dim=-1)
        query_positions = torch.concat([e(query_pos_norm.view(-1, 3)) for e in self.relation_field.encs], dim=-1)
        relation_pre_embd = torch.cat((
            query_positions,
            field_positions
        ), dim=-1)
        rel_feat = self.relation_field.relation_net(relation_pre_embd)

        semantic_dict["relation"] = rel_feat.view(*position_pos.shape[:2], -1)
        return semantic_dict

    @torch.no_grad()
    def get_outputs_for_points_with_query_batch(self, points_batch, query_batch, points_sem_emb=None, query_sem_emb=None) -> Dict[str, torch.Tensor]:
        """
        get point features for given query points,
        points_batch: [N,3] numpy array
        query_batch: [K,3] numpy array
        """
        # map both points and query points to the same shape [N,K,3]
        N, K = points_batch.shape[0], query_batch.shape[0]
        points_batch = np.expand_dims(points_batch, axis=1)
        query_batch = np.expand_dims(query_batch, axis=0)
        points_batch = np.repeat(points_batch, query_batch.shape[1], axis=1)
        query_batch = np.repeat(query_batch, points_batch.shape[0], axis=0)
        # now flatten both points and query points
        points_batch = points_batch.reshape(-1,3)
        query_batch = query_batch.reshape(-1,3)

        # add spatial distortion
        position_pos = torch.from_numpy(points_batch).cuda()
        position_pos_dist = self.relation_field.spatial_distortion(position_pos)
        position_pos_norm = (position_pos_dist+2.)/4.
        # add spatial distortion to query point
        query = query_batch
        query_pos = torch.from_numpy(query).cuda()
        query_pos_dist = self.relation_field.spatial_distortion(query_pos)
        query_pos_norm = (query_pos_dist+2.)/4.
        
        # compute relation features
        field_positions = torch.concat([e(position_pos_norm.view(-1, 3)) for e in self.relation_field.encs], dim=-1)
        query_positions = torch.concat([e(query_pos_norm.view(-1, 3)) for e in self.relation_field.encs], dim=-1)

        if points_sem_emb is not None and query_sem_emb is not None:
            relation_pre_embd = torch.cat((
                query_sem_emb,
                query_positions,
                points_sem_emb,
                field_positions
            ), dim=-1)
        else:
            relation_pre_embd = torch.cat((
                query_positions,
                field_positions
            ), dim=-1)
        rel_feat = self.relation_field.relation_net(relation_pre_embd)

        # rel_feat is of shape [N*K, 512], now reshape again to [N,K,512]
        rel_feat = rel_feat.view(N,K,-1)
        # now lets accumulate features across query points using mean
        rel_feat = rel_feat.mean(dim=1)

        return {
                "relation": rel_feat
        }
    
