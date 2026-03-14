# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from GARField
#   (https://github.com/chungmin99/garfield
# Copyright (c) 2014 GARField authors, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

"""
Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Type, Union

import torch
import numpy as np
import h5py
import os
import os.path as osp

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

from relationfield.relationfield_pixel_sampler import RelationFieldPixelSampler
from relationfield.data.utils.img_group_model import ImgGroupModelConfig, ImgGroupModel
from relationfield.data.utils.openseg_dataloader import OpenSegDataloader
from relationfield.data.utils.gpt_bert_dataloader import GPTDataloader


from rich.progress import Console
CONSOLE = Console(width=120)

def calculate_weight_factor(occurrences, min_occurrences=100, max_occurrences=5000, min_weight=1, max_weight=10):
    occurrences_tensor = occurrences
    clamped_occurrences = torch.clamp(occurrences_tensor, min=min_occurrences, max=max_occurrences)
    weight_factor = min_weight + (max_weight - min_weight) * (1 - (clamped_occurrences - min_occurrences) / (max_occurrences - min_occurrences))
    return weight_factor

@dataclass
class RelationFieldDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: RelationFieldDataManager)
    """The datamanager class to use."""
    img_group_model: ImgGroupModelConfig = ImgGroupModelConfig()
    inverse_relationship: bool = True
    """The SAM model to use. This can be any other model that outputs masks..."""


class RelationFieldDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """
    Tacking on grouping info to the normal VanillaDataManager.
    """

    config: RelationFieldDataManagerConfig
    train_pixel_sampler: Optional[RelationFieldPixelSampler] = None

    def __init__(
        self,
        config: RelationFieldDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        # ns-viewer uses test_mode="test" by default, which triggers full image caching
        # in the base datamanager and can OOM on low-memory systems. Inference mode
        # is sufficient for interactive checkpoint viewing.
        if test_mode == "test":
            test_mode = "inference"

        if os.getenv("NERFACTO_DEPTH"):
            print('---using depth dataset---')
            self.dataset_type = None
            self.dataset_type = DepthDataset
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )
        self.img_group_model: ImgGroupModel = self.config.img_group_model.setup(device=self.device)

        # This is where all the group data + statistics is stored.
        # Note that this can get quite big (~10GB if 300 images, ...)
        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        self.sam_data_path = Path(cache_dir) / "sam_data.hdf5"

        self.pixel_level_keys = None
        self.scale_3d = None
        self.group_cdf = None
        self.scale_3d_statistics = None
        
        # Avoid materializing all training images in memory; only image shape is needed.
        sample_image = self.train_dataset[0]["image"]
        image_shape = list(sample_image.shape[:2])
        openseg_cache_path = Path(osp.join(cache_dir, "openseg.npy"))
        clip_cache_path = Path(osp.join(cache_dir, "clip.npy"))
        gpt_cache_path = Path(osp.join(cache_dir, "gpt.pkl"))
        llama_cache_path = Path(osp.join(cache_dir, "llama.pkl"))
        affordance_cache_path = Path(osp.join(cache_dir, "affordance.pkl"))

        image_pathes = self.train_dataset._dataparser_outputs.image_filenames
        gpt_path = Path(osp.join(self.config.dataparser.data, "chatgpt"))
        llama_path = Path(osp.join(self.config.dataparser.data, "chatllama"))
        affordance_path = Path(osp.join(self.config.dataparser.data, "affordance_chatgpt"))
        
        self.pcd = None
        if 'points3D_xyz' in self.train_dataset._dataparser_outputs.metadata:
            self.pcd = self.train_dataset._dataparser_outputs.metadata['points3D_xyz']
            if self.pcd.shape[0] > 150000:
                mask = np.random.choice(self.pcd.shape[0], 150000, replace=False)
                self.pcd = self.pcd[mask]
            
        
        self.openseg_dataloader = OpenSegDataloader(
            image_list=image_pathes,
            device=self.device,
            cfg={"image_shape": image_shape},
            cache_path=openseg_cache_path,
        )
        
        self.relation_bert_dataloader = GPTDataloader(
            gpt_output_dir=gpt_path,
            device=self.device,
            cfg={"image_shape": image_shape},
            cache_path=gpt_cache_path,
        )
        
        jina_none_path = '/'.join(cache_dir.split('/')[:-1]) + '/jina_none_emb.pt'
        jina_none_embd = torch.load(jina_none_path, weights_only=False)
        if isinstance(jina_none_embd, np.ndarray):
            self.jina_none_embd = torch.from_numpy(jina_none_embd)
        else:
            self.jina_none_embd = torch.as_tensor(jina_none_embd)
        torch.cuda.empty_cache()

    def load_sam_data(self) -> bool:
        """
        Loads the SAM data (masks, 3D scales, etc.) through hdf5.
        If the file doesn't exist, returns False.
        """
        prefix = self.img_group_model.config.model_type
        if osp.exists(self.sam_data_path):
            sam_data = h5py.File(self.sam_data_path, "r")
            if prefix not in sam_data.keys():
                return False

            sam_data = sam_data[prefix]

            pixel_level_keys_list, scales_3d_list, group_cdf_list = [], [], []

            num_entries = len(sam_data["pixel_level_keys"].keys())
            for i in range(num_entries):
                pixel_level_keys_list.append(
                    torch.from_numpy(sam_data["pixel_level_keys"][str(i)][...])
                )
            self.pixel_level_keys = torch.nested.nested_tensor(pixel_level_keys_list)
            del pixel_level_keys_list

            for i in range(num_entries):
                scales_3d_list.append(torch.from_numpy(sam_data["scale_3d"][str(i)][...]))
            self.scale_3d = torch.nested.nested_tensor(scales_3d_list)
            self.scale_3d_statistics = torch.cat(scales_3d_list)
            del scales_3d_list

            for i in range(num_entries):
                group_cdf_list.append(torch.from_numpy(sam_data["group_cdf"][str(i)][...]))
            self.group_cdf = torch.nested.nested_tensor(group_cdf_list)
            del group_cdf_list

            return True

        return False

    def save_sam_data(self, pixel_level_keys, scale_3d, group_cdf):
        """Save the SAM grouping data to hdf5."""
        prefix = self.img_group_model.config.model_type
        # make the directory if it doesn't exist
        if not osp.exists(self.sam_data_path.parent):
            os.makedirs(self.sam_data_path.parent)

        # Append, not overwrite -- in case of multiple runs with different settings.
        with h5py.File(self.sam_data_path, "a") as f:
            for i in range(len(pixel_level_keys)):
                f.create_dataset(f"{prefix}/pixel_level_keys/{i}", data=pixel_level_keys[i])
                f.create_dataset(f"{prefix}/scale_3d/{i}", data=scale_3d[i])
                f.create_dataset(f"{prefix}/group_cdf/{i}", data=group_cdf[i])

    @staticmethod
    def create_pixel_mask_array(masks: torch.Tensor):
        """
        Create per-pixel data structure for grouping supervision.
        pixel_mask_array[x, y] = [m1, m2, ...] means that pixel (x, y) belongs to masks m1, m2, ...
        where Area(m1) < Area(m2) < ... (sorted by area).
        """
        max_masks = masks.sum(dim=0).max().item()
        image_shape = masks.shape[1:]
        pixel_mask_array = torch.full(
            (max_masks, image_shape[0], image_shape[1]), -1, dtype=torch.int
        ).to(masks.device)

        for m, mask in enumerate(masks):
            mask_clone = mask.clone()
            for i in range(max_masks):
                free = pixel_mask_array[i] == -1
                masked_area = mask_clone == 1
                right_index = free & masked_area
                if len(pixel_mask_array[i][right_index]) != 0:
                    pixel_mask_array[i][right_index] = m
                mask_clone[right_index] = 0
        pixel_mask_array = pixel_mask_array.permute(1, 2, 0)

        return pixel_mask_array

    def _calculate_3d_groups(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        point: torch.Tensor,
        max_scale: float = 2.0,
    ):
        """
        Calculate the set of groups and their 3D scale for each pixel, and the cdf.
        Returns:
            - pixel_level_keys: [H, W, max_masks]
            - scale: [num_masks, 1]
            - mask_cdf: [H, W, max_masks]
        max_masks is the maximum number of masks that was assigned to a pixel in the image,
         padded with -1s. mask_cdf does *not* include the -1s.
        Refer to the main paper for more details.
        """
        image_shape = rgb.shape[:2]
        depth = depth.view(-1, 1)  # (H*W, 1)
        point = point.view(-1, 3)  # (H*W, 3)

        def helper_return_no_masks():
            # Fail gracefully when no masks are found.
            # Create dummy data (all -1s), which will be ignored later.
            # See: `get_loss_dict_group` in `relationfield_model.py`
            pixel_level_keys = torch.full(
                (image_shape[0], image_shape[1], 1), -1, dtype=torch.int
            )
            scale = torch.Tensor([0.0]).view(-1, 1)
            mask_cdf = torch.full(
                (image_shape[0], image_shape[1], 1), 1, dtype=torch.float
            )
            return (pixel_level_keys, scale, mask_cdf)
        # Calculate SAM masks
        masks = self.img_group_model((rgb.numpy() * 255).astype(np.uint8))

        # If no masks are found, return dummy data.
        if len(masks) == 0:
            return helper_return_no_masks()

        sam_mask = []
        scale = []

        # For all 2D groups,
        # 1) Denoise the masks (through eroding)
        all_masks = torch.stack(
            # [torch.from_numpy(_["segmentation"]).to(self.device) for _ in masks]
            [torch.from_numpy(_).to(self.device) for _ in masks]
        )
        # erode all masks using 3x3 kernel
        eroded_masks = torch.conv2d(
            all_masks.unsqueeze(1).float(),
            torch.full((3, 3), 1.0).view(1, 1, 3, 3).to("cuda"),
            padding=1,
        )
        eroded_masks = (eroded_masks >= 5).squeeze(1)  # (num_masks, H, W)

        # 2) Calculate 3D scale
        # Don't include groups with scale > max_scale (likely to be too noisy to be useful)
        for i in range(len(masks)):
            curr_mask = eroded_masks[i]
            curr_mask = curr_mask.flatten()
            curr_points = point[curr_mask]
            extent = (curr_points.std(dim=0) * 2).norm()
            if extent.item() < max_scale:
                sam_mask.append(curr_mask.reshape(image_shape))
                scale.append(extent.item())

        # If no masks are found, after postprocessing, return dummy data.
        if len(sam_mask) == 0:
            return helper_return_no_masks()

        sam_mask = torch.stack(sam_mask)  # (num_masks, H, W)
        scale = torch.Tensor(scale).view(-1, 1).to(self.device)  # (num_masks, 1)

        # Calculate "pixel level keys", which is a 2D array of shape (H, W, max_masks)
        # Each pixel has a list of group indices that it belongs to, in order of increasing scale.
        pixel_level_keys = self.create_pixel_mask_array(
            sam_mask
        ).long()  # (H, W, max_masks)

        # Calculate group sampling CDF, to bias sampling towards smaller groups
        # Be careful to not include -1s in the CDF (padding, or unlabeled pixels)
        # Inversely proportional to log of mask size.
        mask_inds, counts = torch.unique(pixel_level_keys, return_counts=True)
        mask_sorted = torch.argsort(counts)
        mask_inds, counts = mask_inds[mask_sorted], counts[mask_sorted]
        counts[0] = 0  # don't include -1
        probs = counts / counts.sum()  # [-1, 0, ...]
        mask_probs = torch.gather(probs, 0, pixel_level_keys.reshape(-1) + 1).view(
            pixel_level_keys.shape
        )
        mask_log_probs = torch.log(mask_probs)
        never_masked = mask_log_probs.isinf()
        mask_log_probs[never_masked] = 0.0
        mask_log_probs = mask_log_probs / (
            mask_log_probs.sum(dim=-1, keepdim=True) + 1e-6
        )
        mask_cdf = torch.cumsum(mask_log_probs, dim=-1)
        mask_cdf[never_masked] = 1.0

        return (pixel_level_keys.cpu(), scale.cpu(), mask_cdf.cpu())


    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)

        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]

        ray_bundle = self.train_ray_generator(ray_indices)
        
        batch["openseg"] = self.openseg_dataloader(ray_indices)

        ray_bundle.metadata["fx"] = self.train_dataset.cameras[0].fx.item()
        ray_bundle.metadata["width"] = self.train_dataset.cameras[0].width.item()
        ray_bundle.metadata["fy"] = self.train_dataset.cameras[0].fy.item()
        ray_bundle.metadata["height"] = self.train_dataset.cameras[0].height.item()
                
        if self.pcd is not None:
            batch["pcd"] = self.pcd
            
        return ray_bundle, batch
    
    def next_rel_map(self, ray_bundle: RayBundle, batch: Dict[str, Any]):
        """_summary_

        Args:
            ray_bundle (RayBundle): _description_
            batch (Dict[str, Any]): _description_
        """
                    
        indices = batch["indices"].long().detach().cpu()
        img_ind = indices[:, 0]
        w,h = ray_bundle.metadata["width"], ray_bundle.metadata["height"]
        
        sample_points = len(img_ind)
        
        n_queries = 1
        query_2d = torch.stack((torch.randint(low=0, high=h, size=(len(img_ind),n_queries,)),torch.randint(low=0, high=w, size=(len(img_ind),n_queries,))),dim=1)
        query_2d = query_2d.permute(2,0,1)

        query_2d = torch.cat((indices[:,0].view(1,-1,1).repeat(n_queries,1,1), query_2d), dim=-1)
        
        rel_batch = self.relation_bert_dataloader(indices, query_2d)
        rel_weight = None
        # subjects are nerf rays, objects are query points, such that we can answer the question "what is the relection of the nerf rays with this point"
        if self.config.inverse_relationship:
            # this is aligned with the tag subject_object_inverse
            rel_pairs = torch.stack((rel_batch['segmentation_map_query'], rel_batch['segmentation_map_class'].unsqueeze(0).repeat(n_queries,1)),dim=-1)
            if "segmentation_count_class" in rel_batch:
                rel_weight = torch.stack((rel_batch['segmentation_count_query'], rel_batch['segmentation_count_class'].unsqueeze(0).repeat(n_queries,1)),dim=-1)
        else:
            rel_pairs = torch.stack((rel_batch['segmentation_map_class'].unsqueeze(0).repeat(n_queries,1), rel_batch['segmentation_map_query']),dim=-1)
            if "segmentation_count_class" in rel_batch:
                rel_weight = torch.stack((rel_batch['segmentation_count_class'].unsqueeze(0).repeat(n_queries,1), rel_batch['segmentation_count_query']),dim=-1)
        

        if rel_weight is None:
            rel_weight = torch.ones((rel_pairs.shape[0],rel_pairs.shape[1],2)).to(self.device)
        else:
            rel_weight = rel_weight.to(self.device)

        rel_embds = []
        for qid in range(n_queries):
            rel_embds.append(torch.stack([rel_batch['rel_embeds'][i].get((rel_pairs[qid,i,0].item(),rel_pairs[qid,i,1].item()),self.jina_none_embd) for i in range(len(rel_pairs[qid]))]).to( device=self.device))
            
        rel_embds = torch.stack(rel_embds)
        mask = ~((rel_embds == self.jina_none_embd.cuda()).all(dim=-1))

        idxs = torch.where(mask, torch.arange(n_queries, device=mask.device).view(-1, 1), torch.tensor(-1, device=mask.device))
        selected_indices, _ = torch.max(idxs, dim=0)

        mask = mask[selected_indices, torch.arange(sample_points, device=mask.device)]
        rel_embds = rel_embds[selected_indices, torch.arange(sample_points, device=mask.device)]
        query_2d = query_2d[selected_indices.cpu(), torch.arange(sample_points)]

        rel_weight = torch.min(rel_weight[selected_indices, torch.arange(sample_points)],dim=-1)[0]
        rel_weight = calculate_weight_factor(rel_weight)
        
        batch["relation_embd"] = rel_embds
        batch["rel_weight"] = rel_weight
        query_bundle = self.train_ray_generator(query_2d)
        batch["query_bundle"] = query_bundle
        batch["query_mask"] = mask
        batch["query_2d"] = query_2d
        
        query_bundle.metadata = dict()
        query_bundle.metadata["fx"] = self.train_dataset.cameras[0].fx.item()
        query_bundle.metadata["width"] = self.train_dataset.cameras[0].width.item()
        query_bundle.metadata["fy"] = self.train_dataset.cameras[0].fy.item()
        query_bundle.metadata["height"] = self.train_dataset.cameras[0].height.item()
        query_bundle.metadata["n_query_rays"] = query_2d.shape[0]
        
        del rel_batch
        
        
    def next_group(self, ray_bundle: RayBundle, batch: Dict[str, Any]):
        """Returns the rays' mask and 3D scales for grouping.
        We add to `batch` the following:
            - "mask_id": [batch_size,]
            - "scale": [batch_size,]
            - "nPxImg": int == `num_rays_per_image`
        This function also adds `scale` to `ray_bundle.metadata`.

        We're using torch nested tensors -- this means that it's difficult to index into them.
        At least now, it seems possible to index normally into a leaf tensor.
        """
        indices = batch["indices"].long().detach().cpu()
        npximg = self.train_pixel_sampler.num_rays_per_image
        img_ind = indices[:, 0]
        x_ind = indices[:, 1]
        y_ind = indices[:, 2]

        # sampled_imgs = img_ind[::npximg]
        mask_id = torch.zeros((indices.shape[0],), device=self.device)
        scale = torch.zeros((indices.shape[0],), device=self.device)

        random_vec_sampling = (torch.rand((1,)) * torch.ones((npximg,))).view(-1, 1)
        random_vec_densify = (torch.rand((1,)) * torch.ones((npximg,))).view(-1, 1)

        for i in range(0, indices.shape[0], npximg):
            img_idx = img_ind[i]

            # Use `random_vec` to choose a group for each pixel.
            per_pixel_index = self.pixel_level_keys[img_idx][
                x_ind[i : i + npximg], y_ind[i : i + npximg]
            ]
            random_index = torch.sum(
                random_vec_sampling.view(-1, 1)
                > self.group_cdf[img_idx][x_ind[i : i + npximg], y_ind[i : i + npximg]],
                dim=-1,
            )

            # `per_pixel_index` encodes the list of groups that each pixel belongs to.
            # If there's only one group, then `per_pixel_index` is a 1D tensor
            # -- this will mess up the future `gather` operations.
            if per_pixel_index.shape[-1] == 1:
                per_pixel_mask = per_pixel_index.squeeze()
            else:
                per_pixel_mask = torch.gather(
                    per_pixel_index, 1, random_index.unsqueeze(-1)
                ).squeeze()
                per_pixel_mask_ = torch.gather(
                    per_pixel_index,
                    1,
                    torch.max(random_index.unsqueeze(-1) - 1, torch.Tensor([0]).int()),
                ).squeeze()

            mask_id[i : i + npximg] = per_pixel_mask.to(self.device)

            # interval scale supervision
            curr_scale = self.scale_3d[img_idx][per_pixel_mask]
            curr_scale[random_index == 0] = (
                self.scale_3d[img_idx][per_pixel_mask][random_index == 0]
                * random_vec_densify[random_index == 0]
            )
            for j in range(1, self.group_cdf[img_idx].shape[-1]):
                if (random_index == j).sum() == 0:
                    continue
                curr_scale[random_index == j] = (
                    self.scale_3d[img_idx][per_pixel_mask_][random_index == j]
                    + (
                        self.scale_3d[img_idx][per_pixel_mask][random_index == j]
                        - self.scale_3d[img_idx][per_pixel_mask_][random_index == j]
                    )
                    * random_vec_densify[random_index == j]
                )
            scale[i : i + npximg] = curr_scale.squeeze().to(self.device)

        batch["mask_id"] = mask_id
        batch["scale"] = scale
        batch["nPxImg"] = npximg
        ray_bundle.metadata["scale"] = batch["scale"]
