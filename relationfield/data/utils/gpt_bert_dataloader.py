# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import gc
import os
import os.path as osp
import glob
import json
import pickle
import numpy as np
from pathlib import Path

import torch
from relationfield.data.utils.gpt_bert_extractor import extract_bert_mask_feature
from tqdm import tqdm


def segment_pixel_count(segmentation_mask):
    unique_segments, counts = np.unique(segmentation_mask, return_counts=True)
    segment_to_count = dict(zip(unique_segments, counts))
    count_mask = np.vectorize(segment_to_count.get)(segmentation_mask)
    return count_mask


class GPTDataloader:
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        gpt_output_dir: str,
        cache_path: str = None,
    ):
        self.cfg = cfg
        self.device = device
        self.cache_path = cache_path
        self.data = None  # only expect data to be cached, nothing else
        self.try_load(gpt_output_dir)  # don't save image_list, avoid duplicates
        self.segmentation_map_tensor = torch.from_numpy(self.data["segmentation_map"])
        self.segmentation_count_tensor = None
        if "segmentation_map_count" in self.data:
            self.segmentation_count_tensor = torch.from_numpy(self.data["segmentation_map_count"])

    def create(self, gpt_output_dir):
        # load jina model
        from transformers import AutoModel

        jina_model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v3", trust_remote_code=True
        ).cuda()
        jina_encode = lambda x: jina_model.encode(
            x, task="text-matching", truncate_dim=512
        )
        torch.save(
            jina_encode("none"),
            os.path.join(Path(self.cache_path).parents[1], "jina_none_emb.pt"),
        )
        # load gpt obj & rel outputs
        masks_paths = glob.glob(osp.join(gpt_output_dir, "*_masks.npy"))
        masks_paths.sort()
        text_outputs_paths = glob.glob(osp.join(gpt_output_dir, "*_gpt_output.txt"))
        text_outputs_paths.sort()
        tag2classes_paths = glob.glob(osp.join(gpt_output_dir, "*_tag2class.json"))
        tag2classes_paths.sort()
        relation_dicts_paths = glob.glob(
            osp.join(gpt_output_dir, "*_relation_dict.json")
        )
        relation_dicts_paths.sort()

        bert_obj_embeds = []
        bert_rel_embeds = []
        bert_seg_maps = []
        bert_segmentation_map = []
        bert_segmentation_count_map = []
        bert_obj2sub = []
        img_feats = []
        for image_id in tqdm(
            range(len(masks_paths)), desc="bert", total=len(masks_paths), leave=False
        ):
            # print(image_id)
            with torch.no_grad():
                mask_path = masks_paths[image_id]
                tag2class_path = tag2classes_paths[image_id]
                relation_dict_path = relation_dicts_paths[image_id]

                h = self.cfg["image_shape"][0] // 4
                w = self.cfg["image_shape"][1] // 4

                (
                    feat_2d,
                    obj_class_embds,
                    rel_embds_dict,
                    seg_maps,
                    segmentation_map,
                    obj2sub,
                ) = extract_bert_mask_feature(
                    mask_path,
                    tag2class_path,
                    relation_dict_path,
                    jina_encode,
                    img_size=[h, w],
                )  # img_size=[240, 320]

            segmentation_count_map = segment_pixel_count(segmentation_map)
            bert_obj_embeds.append(obj_class_embds.cpu().detach())
            bert_rel_embeds.append(rel_embds_dict)
            bert_seg_maps.append(seg_maps)
            bert_segmentation_map.append(segmentation_map)
            bert_segmentation_count_map.append(segmentation_count_map)
            bert_obj2sub.append(obj2sub)
            img_feats.append(feat_2d.cpu().detach())

        del jina_model
        del jina_encode
        gc.collect()
        torch.cuda.empty_cache()
        self.data = {
            "rel_embeds": bert_rel_embeds,
            "segmentation_map": np.stack(bert_segmentation_map),
            "segmentation_map_count": np.stack(bert_segmentation_count_map),
        }

    def try_load(self, gpt_output_path: str):
        try:
            self.load()
        except (FileNotFoundError, ValueError):
            self.create(gpt_output_path)
            self.save()

    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            print("Config mismatch, using saved config")
            # raise ValueError("Config mismatch")
        with open(self.cache_path.with_suffix(".pkl"), "rb") as f:
            self.data = pickle.load(f)
        # Backward-compat for old caches stored as python lists.
        if isinstance(self.data.get("segmentation_map"), list):
            self.data["segmentation_map"] = np.stack(self.data["segmentation_map"])
        if isinstance(self.data.get("segmentation_map_count"), list):
            self.data["segmentation_map_count"] = np.stack(self.data["segmentation_map_count"])

    def save(self):
        os.makedirs(self.cache_path.parent, exist_ok=True)
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
        with open(self.cache_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self.data, f)

    def __call__(self, img_points, query_points):
        # img_points: (B, 3) # (img_ind, x, y)

        img_scale = (
            self.data["segmentation_map"][0].shape[0] / self.cfg["image_shape"][0],
            self.data["segmentation_map"][0].shape[1] / self.cfg["image_shape"][1],
        )
        # masks are actually of the ture shape
        x_ind, y_ind = (
            (img_points[:, 1] * img_scale[0]).long(),
            (img_points[:, 2] * img_scale[1]).long(),
        )
        query_x_ind, query_y_ind = (
            (query_points[..., 1] * img_scale[0]).long(),
            (query_points[..., 2] * img_scale[1]).long(),
        )

        batch_idx = img_points[:, 0].long()
        # get indices of tmp where tmp is not 0

        outdict = {
            "rel_embeds": [self.data["rel_embeds"][i] for i in batch_idx],
            "segmentation_map_class": self.segmentation_map_tensor[
                img_points[:, 0].long(), x_ind, y_ind
            ],
            "segmentation_map_query": self.segmentation_map_tensor[
                query_points[..., 0].long(), query_x_ind, query_y_ind
            ],
        }
        if self.segmentation_count_tensor is not None:
            outdict["segmentation_count_class"] = self.segmentation_count_tensor[
                img_points[:, 0].long(), x_ind, y_ind
            ]
            outdict["segmentation_count_query"] = self.segmentation_count_tensor[
                query_points[..., 0].long(), query_x_ind, query_y_ind
            ]
        return outdict
