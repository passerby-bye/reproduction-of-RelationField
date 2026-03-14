# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from GARField
#   (https://github.com/chungmin99/garfield)
# Copyright (c) 2014 GARField authors, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)

from relationfield.relationfield_pipeline import RelationFieldPipelineConfig
from relationfield.relationfield_datamanager import RelationFieldDataManagerConfig
from relationfield.relationfield_pixel_sampler import RelationFieldPixelSamplerConfig
from relationfield.relationfield_model import RelationFieldModelConfig
from relationfield.instance_field import GarFieldConfig
from relationfield.data.utils.img_group_model import ImgGroupModelConfig

try:
    from relationfield.relationfield_gaussian_pipeline import RelationFieldGaussianPipelineConfig
except ModuleNotFoundError:
    RelationFieldGaussianPipelineConfig = None


relationfield_method = MethodSpecification(
    config=TrainerConfig(
        method_name="relationfield",
        steps_per_eval_image=100,
        steps_per_eval_batch=100,
        steps_per_save=2000,
        steps_per_eval_all_images=100000,
        max_num_iterations=30000,
        mixed_precision=True,
        use_grad_scaler=True,
        pipeline=RelationFieldPipelineConfig(
            datamanager=RelationFieldDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99, load_3D_points=True),
                train_num_rays_per_batch=1024, # reduced for low-memory training
                eval_num_rays_per_batch=1024, # reduced for low-memory training
                pixel_sampler=RelationFieldPixelSamplerConfig(
                    num_rays_per_image=64,  # 1024/64 = 16 images per batch
                ),
                img_group_model=ImgGroupModelConfig(
                    model_type="sam_hf",  
                    # Can choose out of "sam_fb", "sam_hf", "maskformer"
                    # Used sam_fb for the paper, see `img_group_model.py`. 
                    device="cuda",
                ),
            ),
            model=RelationFieldModelConfig(
                instance_field=GarFieldConfig(
                    n_instance_dims=256  # 256 in original
                ),
                eval_num_rays_per_chunk=1 << 11,
                # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(12, 12),
                hashgrid_resolutions=((16, 128), (128, 512)),
                num_feat_samples=12,
                num_semantic_samples=12,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "relationfield": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=30000, warmup_steps=2000),  
            },
            'camera_opt': {
                'optimizer': AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                'scheduler': ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000)
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 11),
        vis="viewer",
    ),
    description="NeRF with Relation Field",
)


if RelationFieldGaussianPipelineConfig is not None:
    relationfield_gauss_method = MethodSpecification(
        config=TrainerConfig(
            method_name="relationfield-gauss",
            steps_per_eval_image=100,
            steps_per_eval_batch=100,
            steps_per_save=2000,
            steps_per_eval_all_images=100000,
            max_num_iterations=30000,
            mixed_precision=False,
            gradient_accumulation_steps={'camera_opt': 100, 'color': 10, 'shs': 10},
            pipeline=RelationFieldGaussianPipelineConfig(
                datamanager=FullImageDatamanagerConfig(
                    dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                    cache_images_type="uint8",
                ),
                model=SplatfactoModelConfig(),
            ),
            optimizers={
                "means": {
                    "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                    "scheduler": ExponentialDecaySchedulerConfig(
                        lr_final=1.6e-6,
                        max_steps=30000,
                    ),
                },
                "features_dc": {
                    "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                    "scheduler": None,
                },
                "features_rest": {
                    "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                    "scheduler": None,
                },
                "opacities": {
                    "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                    "scheduler": None,
                },
                "scales": {
                    "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                    "scheduler": None,
                },
                "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
                "camera_opt": {
                    "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                    "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
                },
                "bilateral_grid": {
                    "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                    "scheduler": ExponentialDecaySchedulerConfig(
                        lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                    ),
                },
            },
            viewer=ViewerConfig(num_rays_per_chunk=1 << 11),
            vis="viewer",
        ),
        description="Gaussian Splatting with Relation Field",
    )
