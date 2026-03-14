# This source code is from OpenNerf
#   (https://github.com/opennerf/opennerf
# Copyright (c) 2024 OpenNerf authors
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import gc
import os
import typing

import numpy as np
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import torch
from relationfield.data.utils.feature_dataloader import FeatureDataloader
from relationfield.data.utils.openseg_extractor import extract_openseg_img_feature
from tqdm import tqdm


class OpenSegDataloader(FeatureDataloader):

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
    ):
        assert "image_shape" in cfg
        super().__init__(cfg, device, image_list, cache_path)
    
    def create(self, image_path_list):
        # TensorFlow OpenSeg often fails on mixed CUDA/cuDNN setups.
        # Keep OpenSeg on CPU by default; Torch training still uses GPU.
        force_cpu = os.environ.get("RELATIONFIELD_OPENSEG_CPU", "1") == "1"
        if force_cpu:
            try:
                tf2.config.set_visible_devices([], "GPU")
            except Exception:
                # Device visibility can be immutable after TF runtime init.
                pass

        saved_model_path = 'models/openseg_exported_clip'
        saved_model_path = os.path.realpath(os.path.expanduser(saved_model_path))
        openseg_model = tf2.saved_model.load(saved_model_path, tags=[tf.saved_model.tag_constants.SERVING],)

        num_images = len(image_path_list)
        openseg_embeds = None
        for image_id in tqdm(range(num_images), desc='openseg', total=num_images, leave=False):
            with torch.no_grad():
                image_path = image_path_list[image_id]
                h = self.cfg['image_shape'][0] // 4
                w = self.cfg['image_shape'][1] // 4
                
                descriptors = extract_openseg_img_feature(image_path, openseg_model, img_size=[h, w])  # img_size=[240, 320]
            descriptors = descriptors.permute(1, 2, 0).cpu().numpy()
            if openseg_embeds is None:
                openseg_embeds = np.empty((num_images, *descriptors.shape), dtype=np.float16)
            openseg_embeds[image_id] = descriptors.astype(np.float16, copy=False)

        del openseg_model
        gc.collect()
        torch.cuda.empty_cache()
        self.data = torch.from_numpy(openseg_embeds)

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)
