# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from GARField
#   (https://github.com/chungmin99/garfield
# Copyright (c) 2014 GARField authors, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional, List, Dict
from pathlib import Path
import trimesh
import viser
import viser.transforms as vtf
import open3d as o3d
import cv2
import numpy as np
import time
import torch
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from torch.cuda.amp.grad_scaler import GradScaler
from nerfstudio.viewer.viewer_elements import *
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.models.splatfacto import SplatfactoModel

from cuml.cluster.hdbscan import HDBSCAN
from nerfstudio.utils.colormaps import apply_pca_colormap
from nerfstudio.models.splatfacto import RGB2SH

import matplotlib
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors


from relationfield.relationfield_pipeline import RelationFieldPipeline
from relationfield.type_aliases import TensorType


def generate_random_colors(N=5000) -> torch.Tensor:
    """Generate random colors for visualization"""
    hs = np.random.uniform(0, 1, size=(N, 1))
    ss = np.random.uniform(0.6, 0.61, size=(N, 1))
    vs = np.random.uniform(0.84, 0.95, size=(N, 1))
    hsv = np.concatenate([hs, ss, vs], axis=-1)
    # convert to rgb
    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8)[None, ...], cv2.COLOR_HSV2RGB)
    return torch.Tensor(rgb.squeeze() / 255.0)


@dataclass
class RelationFieldGaussianPipelineConfig(VanillaPipelineConfig):
    """Gaussian Splatting, but also loading GARField grouping field from ckpt."""
    _target: Type = field(default_factory=lambda: RelationFieldGaussianPipeline)
    relationfield_ckpt: Optional[Path] = None  # Need to specify this


class RelationFieldGaussianPipeline(VanillaPipeline):
    """
    Trains a Gaussian Splatting model, but also loads a GARField grouping field from ckpt.
    This grouping field allows you to:
     - interactive click-based group selection (you can drag it around)
     - scene clustering, then group selection (also can drag it around)

    Note that the pipeline training must be stopped before you can interact with the scene!!
    """
    model: SplatfactoModel
    relationfield_pipeline: List[RelationFieldPipeline]  # To avoid importing Viewer* from nerf pipeline
    state_stack: List[Dict[str, TensorType]]  # To revert to previous state
    crop_group_list: List[TensorType]  # For storing gaussian crops (based on click point)
    crop_transform_handle: Optional[viser.TransformControlsHandle]  # For storing scene transform handle -- drag!
    cluster_labels: Optional[TensorType]  # For storing cluster labels

    def __init__(
        self,
        config: RelationFieldGaussianPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: typing.Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)

        print("Loading relation feature model...")
        assert config.relationfield_ckpt is not None, "Need to specify relationfield checkpoint"
        from nerfstudio.utils.eval_utils import eval_setup
        _, relationfield_pipeline, _, _ = eval_setup(
            config.relationfield_ckpt, test_mode="inference"
        )
        self.relationfield_pipeline = [relationfield_pipeline]
        self.state_stack = []

        self.colormap = generate_random_colors()
        self.clip_model = None
        self.jina_encode = None
        self.viewer_control = ViewerControl()

        self.a_interaction_method = ViewerDropdown(
            "Interaction Method",
            default_value="Interactive",
            options=["Interactive", "Clustering"],
            cb_hook=self._update_interaction_method
        )

        
        self.click_gaussian_relation: ViewerButton = ViewerButton(
            name="Relation point", cb_hook=self._click_gaussian_relation, visible=True
        )
        self.relation_click_location = None
        self.relation_click_handle = None

        self.positives = []
        self.negatives =  ["object", "things", "stuff", "texture", "other","nothing", "empty", "texture", "photo", "image", "picture"]
        self.object_query_text = ViewerText("Object Query", "", cb_hook=self._object_query)

        self.relation_positives = []
        self.negatives =  ["object", "things", "stuff", "texture", "other","nothing", "empty", "texture", "photo", "image", "picture"]
        self.relation_negatives_general = ["none","next to","and"]
        self.relation_negatives = ["none","next to","and"]
        self.relationship_query_text = ViewerText("Relationship Query", "", cb_hook=self._relation_query)

        self.cluster_scene = ViewerButton(name="Cluster Scene", cb_hook=self._cluster_scene, disabled=False, visible=False)
        self.segment_scene = ViewerButton(name="Segment Scene", cb_hook=self._segment_scene, disabled=False, visible=False)
        self.cluster_scene_scale = ViewerSlider(name="Cluster Scale", min_value=0.0, max_value=2.0, step=0.01, default_value=0.0, disabled=False, visible=False)
        self.cluster_scene_shuffle_colors = ViewerButton(name="Reshuffle Cluster Colors", cb_hook=self._reshuffle_cluster_colors, disabled=False, visible=False)
        self.cluster_labels = None

        self.reset_state = ViewerButton(name="Reset State", cb_hook=self._reset_state, disabled=True)

        self.z_export_options = ViewerCheckbox(name="Export Options", default_value=False, cb_hook=self._update_export_options)
        self.z_export_options_visible_gaussians = ViewerButton(
            name="Export Visible Gaussians",
            visible=False,
            cb_hook=self._export_visible_gaussians
            )
        self.z_export_options_camera_path_filename = ViewerText("Camera Path Filename", "", visible=False)
        self.z_export_options_camera_path_render = ViewerButton("Render Current Pipeline", cb_hook=self.render_from_path, visible=False)

    def _update_interaction_method(self, dropdown: ViewerDropdown):
        """Update the UI based on the interaction method"""
        hide_in_interactive = (not (dropdown.value == "Interactive")) # i.e., hide if in interactive mode

        self.cluster_scene.set_hidden((not hide_in_interactive))
        self.segment_scene.set_hidden((not hide_in_interactive))
        self.cluster_scene_scale.set_hidden((not hide_in_interactive))
        self.cluster_scene_shuffle_colors.set_hidden((not hide_in_interactive))

        self.click_gaussian_relation.set_hidden(hide_in_interactive)
        self.object_query_text.set_hidden(hide_in_interactive)
        self.relationship_query_text.set_hidden(hide_in_interactive)
        

    def _update_export_options(self, checkbox: ViewerCheckbox):
        """Update the UI based on the export options"""
        self.z_export_options_camera_path_filename.set_hidden(not checkbox.value)
        self.z_export_options_camera_path_render.set_hidden(not checkbox.value)
        self.z_export_options_visible_gaussians.set_hidden(not checkbox.value)

    def _reset_state(self, button: ViewerButton):
        """Revert to previous saved state"""
        assert len(self.state_stack) > 0, "No previous state to revert to"
        prev_state = self.state_stack.pop()
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name]

        self.relation_click_location = None
        if self.relation_click_handle is not None:
            self.relation_click_handle.remove()
        self.relation_click_handle = None

        self.click_gaussian_relation.set_disabled(False)

        if len(self.state_stack) == 0:
            self.reset_state.set_disabled(True)

        self.cluster_labels = None
        self.cluster_scene.set_disabled(False)
        self.segment_scene.set_disabled(False)
        

    def _queue_state(self):
        """Save current state to stack"""
        import copy
        self.state_stack.append(copy.deepcopy({k:v.detach() for k,v in self.model.gauss_params.items()}))
        self.reset_state.set_disabled(False)

    def _object_query(self,element):
        if len(self.state_stack) == 0:
            self._queue_state()
        self.set_positives(element.value.split(";"))

    def _relation_query(self,element):
        print('---')
        if len(self.state_stack) == 0:
            self._queue_state()
        import time
        tick = time.time()
        self.set_relation_positives(element.value.split(";"))
        print('Time:', time.time()-tick)

    def set_positives(self, text_list):
        self.positives = text_list

        if self.clip_model is None:
            import open_clip
            self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name='ViT-L-14-336', pretrained="openai", device="cuda")
            self.clip_tokenizer = open_clip.get_tokenizer(model_name='ViT-L-14-336')
        
        with torch.no_grad():
            tok_phrases = self.clip_tokenizer(self.positives).to("cuda")
            tok_negatives = self.clip_tokenizer(self.negatives).to("cuda")
            self.pos_embeds = self.clip_model.encode_text(tok_phrases)
            self.neg_embeds = self.clip_model.encode_text(tok_negatives)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)
        self.get_relevancy()

    def set_relation_positives(self, text_list):
        self.relation_positives = text_list
        
        if self.jina_encode is None:
            self.jina_encode =  lambda x: self.relationfield_pipeline[0].model.click_scene.jina_model.encode(x, task='text-matching', truncate_dim=512)
        
        with torch.no_grad():
            
            self.relation_pos_embeds = torch.from_numpy(self.jina_encode(self.relation_positives)).to("cuda")
            self.relation_neg_embeds = torch.from_numpy(self.jina_encode(self.relation_negatives)).to("cuda")
        self.relation_pos_embeds /= self.relation_pos_embeds.norm(dim=-1, keepdim=True)
        self.relation_neg_embeds /= self.relation_neg_embeds.norm(dim=-1, keepdim=True)
        if self.relation_click_location is not None:
            self.get_relation_relevancy()
        

    def get_relevancy(self,) -> torch.Tensor:

        segmentation_model = self.relationfield_pipeline[0].model
        
        positions = self.model.gauss_params['means'].detach()
        embed = segmentation_model.get_segmentation_at_points(positions)  # (N, 256)
        positions = positions.cpu().numpy()

        if self.pos_embeds is None or self.neg_embeds is None or self.clip_model is None:
            return None

        positive_id = 0
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        self.color_gaussian_activation(torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ])

    def get_relation_relevancy(self,) -> torch.Tensor:

        relation_model = self.relationfield_pipeline[0].model
        
        positions = self.model.gauss_params['means'].detach()
        embed = relation_model.get_relation_at_points(positions, self.relation_click_location)  # (N, 256)
        print("N Guassian: ", embed.shape[0])
        positions = positions.cpu().numpy()

        if self.relation_pos_embeds is None or self.relation_neg_embeds is None:
            return None
        positive_id = 0
        phrases_embeds = torch.cat([self.relation_pos_embeds, self.relation_neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.relation_positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.relation_negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        self.color_gaussian_activation(torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.relation_negatives), 2))[
            :, 0, :
        ])

    def color_gaussian_activation(self, activation):
        prev_state = self.state_stack.pop()
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name]
        self._queue_state()
        features_dc = self.model.gauss_params['features_dc'].detach()
        features_rest = self.model.gauss_params['features_rest'].detach()

        if activation is None:
            return 
        activation = activation[:,0]
        p = torch.clip(activation - 0.5, 0, 1).squeeze()
        p = p / (p.max()+1e-6)
        p = torch.clip(p, 0, 0.85) # top 15 percentile is too dark
        overlay = RGB2SH(torch.tensor(matplotlib.colormaps["turbo"].colors, device=p.device)[(p*255).to(torch.long)])


        mask = (p <= 0).squeeze()
        alpha = 0.2
        black_image = torch.zeros_like(overlay).to(overlay.device)

        overlay_dc = 0.9 * overlay  + 0.1 * features_dc
        overlay_rest = 0.9 * torch.zeros_like(features_rest)  + 0.1 * features_rest

        overlay_dc[mask] = (1 - alpha) * features_dc[mask] + alpha * black_image[mask]
        overlay_rest[mask] = (1 - alpha) * features_rest[mask] + alpha * torch.zeros_like(features_rest)[mask]


        opacities = self.model.gauss_params['opacities'].detach()
        self.model.gauss_params['opacities'] = torch.nn.Parameter(opacities.float())


        self.model.gauss_params['features_dc'] = torch.nn.Parameter(overlay_dc)
        self.model.gauss_params['features_rest'] = torch.nn.Parameter(overlay_rest)


        self.viewer_control.viewer._trigger_rerender()  
        
    def _click_gaussian_relation(self, button: ViewerButton):
        """Start listening for click-based 3D point specification.
        Refer to relationfield_interaction.py for more details."""
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_relationclick(click)
            self.click_gaussian_relation.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)

        self.click_gaussian_relation.set_disabled(True)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)


    def _on_relationclick(self, click: ViewerClick):
        """On click, calculate the 3D position of the click and visualize it.
        Refer to relationfield_interaction.py for more details."""

        cam = self.viewer_control.get_camera(500, None, 0)
        cam2world = cam.camera_to_worlds[0, :3, :3]
        import viser.transforms as vtf

        x_pi = vtf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)
        world2cam = (cam2world @ x_pi).inverse()
        # rotate the ray around into cam coordinates
        newdir = world2cam @ torch.tensor(click.direction).unsqueeze(-1)
        z_dir = newdir[2].item()
        # project it into coordinates with matrix
        K = cam.get_intrinsics_matrices()[0]
        coords = K @ newdir
        coords = coords / coords[2]
        pix_x, pix_y = int(coords[0]), int(coords[1])
        self.model.eval()
        outputs = self.model.get_outputs(cam.to(self.device))
        self.model.train()
        with torch.no_grad():
            depth = outputs["depth"][pix_y, pix_x].cpu().numpy()

        self.relation_click_location = np.array(click.origin) + np.array(click.direction) * (depth / z_dir)

        sphere_mesh = trimesh.creation.icosphere(radius=0.2)
        sphere_mesh.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)  # type: ignore
        self.click_handle = None 
        print('click')
        # self.viewer_control.viser_server.add_mesh_trimesh(
        #     name=f"/click",
        #     mesh=sphere_mesh,
        #     position=VISER_NERFSTUDIO_SCALE_RATIO * self.relation_click_location,
        # )

    def _reshuffle_cluster_colors(self, button: ViewerButton):
        """Reshuffle the cluster colors, if clusters defined using `_cluster_scene`."""
        if self.cluster_labels is None:
            return
        self.cluster_scene_shuffle_colors.set_disabled(True)  # Disable user from reshuffling colors
        self.colormap = generate_random_colors()
        colormap = self.colormap

        labels = self.cluster_labels

        features_dc = self.model.gauss_params['features_dc'].detach()
        features_rest = self.model.gauss_params['features_rest'].detach()
        for c_id in range(0, labels.max().int().item() + 1):
            # set the colors of the gaussians accordingly using colormap from matplotlib
            cluster_mask = np.where(labels == c_id)
            features_dc[cluster_mask] = RGB2SH(colormap[c_id, :3].to(self.model.gauss_params['features_dc']))
            features_rest[cluster_mask] = 0

        self.model.gauss_params['features_dc'] = torch.nn.Parameter(self.model.gauss_params['features_dc'])
        self.model.gauss_params['features_rest'] = torch.nn.Parameter(self.model.gauss_params['features_rest'])
        self.cluster_scene_shuffle_colors.set_disabled(False)

    def _cluster_scene(self, button: ViewerButton):
        """Cluster the scene, and assign gaussian colors based on the clusters.
        Also populates self.crop_group_list with the clusters group indices."""

        self._queue_state()  # Save current state
        self.cluster_scene.set_disabled(True)  # Disable user from clustering, while clustering

        scale = self.cluster_scene_scale.value
        grouping_model = self.relationfield_pipeline[0].model
        
        positions = self.model.gauss_params['means'].detach()
        group_feats = grouping_model.get_grouping_at_points(positions, scale).cpu().numpy()  # (N, 256)
        positions = positions.cpu().numpy()

        start = time.time()

        # Cluster the gaussians using HDBSCAN.
        # We will first cluster the downsampled gaussians, then 
        #  assign the full gaussians to the spatially closest downsampled gaussian.

        vec_o3d = o3d.utility.Vector3dVector(positions)
        pc_o3d = o3d.geometry.PointCloud(vec_o3d)
        min_bound = np.clip(pc_o3d.get_min_bound(), -1, 1)
        max_bound = np.clip(pc_o3d.get_max_bound(), -1, 1)
        # downsample size to be a percent of the bounding box extent
        downsample_size = 0.01 * scale
        pc, _, ids = pc_o3d.voxel_down_sample_and_trace(
            max(downsample_size, 0.0001), min_bound, max_bound
        )
        if len(ids) > 1e6:
            print(f"Too many points ({len(ids)}) to cluster... aborting.")
            print( "Consider using interactive select to reduce points before clustering.")
            print( "Are you sure you want to cluster? Press y to continue, else return.")
            # wait for input to continue, if yes then continue, else return
            if input() != "y":
                self.cluster_scene.set_disabled(False)
                return

        id_vec = np.array([points[0] for points in ids])  # indices of gaussians kept after downsampling
        group_feats_downsampled = group_feats[id_vec]
        positions_downsampled = np.array(pc.points)

        print(f"Clustering {group_feats_downsampled.shape[0]} gaussians... ", end="", flush=True)

        # Run cuml-based HDBSCAN
        clusterer = HDBSCAN(
            cluster_selection_epsilon=0.1,
            min_samples=30,
            min_cluster_size=30,
            allow_single_cluster=True,
        ).fit(group_feats_downsampled)

        non_clustered = np.ones(positions.shape[0], dtype=bool)
        non_clustered[id_vec] = False
        labels = clusterer.labels_.copy()
        clusterer.labels_ = -np.ones(positions.shape[0], dtype=np.int32)
        clusterer.labels_[id_vec] = labels

        # Assign the full gaussians to the spatially closest downsampled gaussian, with scipy NearestNeighbors.
        positions_np = positions[non_clustered]
        if positions_np.shape[0] > 0:  # i.e., if there were points removed during downsampling
            k = 1
            nn_model = NearestNeighbors(
                n_neighbors=k, algorithm="auto", metric="euclidean"
            ).fit(positions_downsampled)
            _, indices = nn_model.kneighbors(positions_np)
            clusterer.labels_[non_clustered] = labels[indices[:, 0]]

        labels = clusterer.labels_
        print(f"done. Took {time.time()-start} seconds. Found {labels.max() + 1} clusters.")

        noise_mask = labels == -1
        if noise_mask.sum() != 0 and (labels>=0).sum() > 0:
            # if there is noise, but not all of it is noise, relabel the noise
            valid_mask = labels >=0
            valid_positions = positions[valid_mask]
            k = 1
            nn_model = NearestNeighbors(
                n_neighbors=k, algorithm="auto", metric="euclidean"
            ).fit(valid_positions)
            noise_positions = positions[noise_mask]
            _, indices = nn_model.kneighbors(noise_positions)
            # for now just pick the closest cluster
            noise_relabels = labels[valid_mask][indices[:, 0]]
            labels[noise_mask] = noise_relabels
            clusterer.labels_ = labels

        labels = clusterer.labels_

        colormap = self.colormap

        opacities = self.model.gauss_params['opacities'].detach()
        opacities[labels < 0] = -100  # hide unclustered gaussians
        self.model.gauss_params['opacities'] = torch.nn.Parameter(opacities.float())

        self.cluster_labels = torch.Tensor(labels)
        features_dc = self.model.gauss_params['features_dc'].detach()
        features_rest = self.model.gauss_params['features_rest'].detach()
        for c_id in range(0, labels.max() + 1):
            # set the colors of the gaussians accordingly using colormap from matplotlib
            cluster_mask = np.where(labels == c_id)
            features_dc[cluster_mask] = RGB2SH(colormap[c_id, :3].to(self.model.gauss_params['features_dc']))
            features_rest[cluster_mask] = 0

        self.model.gauss_params['features_dc'] = torch.nn.Parameter(self.model.gauss_params['features_dc'])
        self.model.gauss_params['features_rest'] = torch.nn.Parameter(self.model.gauss_params['features_rest'])

        self.cluster_scene.set_disabled(False)
        self.viewer_control.viewer._trigger_rerender()  # trigger viewer rerender
        
        
    def _segment_scene(self, button: ViewerButton):
        """Cluster the scene, and assign gaussian colors based on the clusters.
        Also populates self.crop_group_list with the clusters group indices."""

        self._queue_state()  # Save current state
        self.segment_scene.set_disabled(True)  # Disable user from clustering, while clustering

        segmentation_model = self.relationfield_pipeline[0].model
        
        positions = self.model.gauss_params['means'].detach()
        segment_feats = segmentation_model.get_segmentation_at_points(positions).cpu().numpy()  # (N, 256)
        positions = positions.cpu().numpy()

        start = time.time()

        # Cluster the gaussians using HDBSCAN.
        # We will first cluster the downsampled gaussians, then 
        #  assign the full gaussians to the spatially closest downsampled gaussian.

        vec_o3d = o3d.utility.Vector3dVector(positions)
        pc_o3d = o3d.geometry.PointCloud(vec_o3d)
        min_bound = np.clip(pc_o3d.get_min_bound(), -1, 1)
        max_bound = np.clip(pc_o3d.get_max_bound(), -1, 1)
        # downsample size to be a percent of the bounding box extent
        downsample_size = 0.0001
        pc, _, ids = pc_o3d.voxel_down_sample_and_trace(
            max(downsample_size, 0.0001), min_bound, max_bound
        )
        if len(ids) > 1e6:
            print(f"Too many points ({len(ids)}) to cluster... aborting.")
            print( "Consider using interactive select to reduce points before clustering.")
            print( "Are you sure you want to cluster? Press y to continue, else return.")
            # wait for input to continue, if yes then continue, else return
            if input() != "y":
                self.segment_scene.set_disabled(False)
                return
        
        id_vec = np.array([points[0] for points in ids])  # indices of gaussians kept after downsampling
        segment_feats_downsampled = segment_feats[id_vec]
        positions_downsampled = np.array(pc.points)

        print(f"PCA {segment_feats_downsampled.shape[0]} gaussians... ", end="", flush=True)

        # Run PCA projection
        segment_colors_downsampled = apply_pca_colormap(torch.from_numpy(segment_feats_downsampled).cuda().float()).cpu().numpy()

        print(f"done. Took {time.time()-start} seconds. ")
        
        segment_colors = -np.ones(positions.shape, dtype=np.float32)
        segment_colors[id_vec] = segment_colors_downsampled

        non_segmented = np.ones(positions.shape[0], dtype=bool)
        non_segmented[id_vec] = False
        positions_np = positions[non_segmented]
        if positions_np.shape[0] > 0:  # i.e., if there were points removed during downsampling
            k = 1
            nn_model = NearestNeighbors(
                n_neighbors=k, algorithm="auto", metric="euclidean"
            ).fit(positions_downsampled)
            _, indices = nn_model.kneighbors(positions_np)
            segment_colors[non_segmented] = segment_colors_downsampled[indices[:, 0]]


        opacities = self.model.gauss_params['opacities'].detach()
        self.model.gauss_params['opacities'] = torch.nn.Parameter(opacities.float())

        features_dc = RGB2SH(torch.from_numpy(segment_colors).to(self.model.gauss_params['features_dc']))
        features_rest = self.model.gauss_params['features_rest']*0
        
        self.model.gauss_params['features_dc'] = torch.nn.Parameter(features_dc)
        self.model.gauss_params['features_rest'] = torch.nn.Parameter(features_rest)

        self.segment_scene.set_disabled(False)
        self.viewer_control.viewer._trigger_rerender()  # trigger viewer rerender

    def _export_visible_gaussians(self, button: ViewerButton):
        """Export the visible gaussians to a .ply file"""
        # location to save
        output_dir = f"outputs/{self.datamanager.config.dataparser.data.name}"
        filename = Path(output_dir) / f"gaussians.ply"
        print("Exporting visible gaussians to ", filename)

        # Copied from exporter.py
        from collections import OrderedDict
        map_to_tensors = OrderedDict()
        model=self.model

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            count = positions.shape[0]
            n = count
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                map_to_tensors["colors"] = (colors * 255).astype(np.uint8)

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)
        from nerfstudio.scripts.exporter import ExportGaussianSplat
        ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)


    def render_from_path(self, button: ViewerButton):
        from nerfstudio.cameras.camera_paths import get_path_from_json
        import json
        from nerfstudio.scripts.render import _render_trajectory_video

        assert self.z_export_options_camera_path_filename.value != ""
        camera_path_filename = Path(self.z_export_options_camera_path_filename.value)
        
        with open(camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        seconds = camera_path["seconds"]
        camera_path = get_path_from_json(camera_path)
        self.model.eval()
        with torch.no_grad():
            _render_trajectory_video(
                self,
                camera_path,
                output_filename=Path('render.mp4'),
                rendered_output_names=['rgb'],
                rendered_resolution_scaling_factor=1.0 ,
                seconds=seconds,
                output_format="video",
            )
        self.model.train()
