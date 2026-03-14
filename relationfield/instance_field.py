# This source code is from GARField 
#   (https://github.com/chungmin99/garfield
# Copyright (c) 2024 GARField authors
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

from enum import Enum
from typing import Dict, Tuple, Callable, Any, Type
from dataclasses import dataclass, field

import numpy as np
import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.base_field import Field
from relationfield.type_aliases import TensorType

try:
    import tinycudann as tcnn
except ImportError:
    pass


class GarFieldHeadNames(Enum):
    """Possible field outputs"""

    INSTANCE = "instance"
    HASHGRID = "hashgrid"


@dataclass
class GarFieldConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: GarField)
    """The field class to instantiate."""

    n_instance_dims: int = 256
    hashgrid_cfg: Dict[str, Any] = field(
        default_factory=lambda: {
            "resolution_range": [(16, 256), (256, 2048)],
            "level": [12, 12],
        }
    )
    """Field parameters. """

    use_single_scale: bool = False
    """For single-scale ablation. For full relationfield, set to False."""


class GarField(Field):
    quantile_transformer: Callable[[TensorType], TensorType]
    config: GarFieldConfig

    def __init__(
        self,
        config: GarFieldConfig,
    ):
        super().__init__()
        self.config = config
        self.spatial_distortion: SceneContraction = SceneContraction()
        self.use_single_scale = self.config.use_single_scale
        hashgrid_cfg = self.config.hashgrid_cfg
        instance_n_dims = self.config.n_instance_dims
        use_single_scale = self.config.use_single_scale

        # This is a trick to make the hashgrid encoding work with the TCNN library.
        self.enc_list = torch.nn.ModuleList(
            [
                self._get_encoding(
                    hashgrid_cfg["resolution_range"][i], hashgrid_cfg["level"][i]
                )
                for i in range(len(hashgrid_cfg["level"]))
            ]
        )
        tot_out_dims = sum([e.n_output_dims for e in self.enc_list])

        # This is the MLP that takes the hashgrid encoding as input.
        # Note the +1 for the scale input.
        self.instance_net = tcnn.Network(
            n_input_dims=tot_out_dims + (0 if use_single_scale else 1),
            n_output_dims=instance_n_dims,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 4,
            },
        )
        self.quantile_transformer = None  # for scale normalization

    @staticmethod
    def _get_encoding(
        res_range: Tuple[int, int], levels: int, indim=3, hash_size=19
    ) -> tcnn.Encoding:
        """
        Helper function to create a HashGrid encoding.
        """
        start_res, end_res = res_range
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc

    def get_outputs(self, ray_samples: RaySamples, scale) -> Dict[FieldHeadNames, TensorType]:
        # x = self.get_hash(ray_samples)
        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        

        xs = [e(positions.view(-1, 3)) for e in self.enc_list]
        x = torch.concat(xs, dim=-1)
        # x = x.view(*ray_samples.frustums.shape, -1)
        x = x / x.norm(dim=-1, keepdim=True)
        instance_scale = torch.ones((x.shape[0], 1), device=x.device) * scale
        instance_pass = self.get_mlp(x, instance_scale.to(x.device)).view(*ray_samples.frustums.shape, -1)
        
        return {
            GarFieldHeadNames.INSTANCE: instance_pass,
        }

    def get_hash(self, ray_samples: RaySamples) -> TensorType:
        """Get the hashgrid encoding. Note that this function does *not* normalize the hash values."""
        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        

        xs = [e(positions.view(-1, 3)) for e in self.enc_list]
        x = torch.concat(xs, dim=-1)
        hash = x.view(*ray_samples.frustums.shape, -1)
        
        return hash

    def get_mlp(self, hash: TensorType, instance_scales: TensorType) -> TensorType:
        """
        Get the relationfield affinity field outputs. Note that this is scale-conditioned.
        This function *does* assume that the hash values are normalized.
        The MLP output is normalized to unit length.
        """
        assert self.quantile_transformer is not None

        # Check that # of rays is the same as # of scales
        assert hash.shape[0] == instance_scales.shape[0]
        epsilon = 1e-5
        if self.use_single_scale:
            instance_pass = self.instance_net(hash)
            return instance_pass / (instance_pass.norm(dim=-1, keepdim=True) + epsilon)

        scales = instance_scales.contiguous().view(-1, 1)

        # Normalize scales before passing to MLP
        scales = self.quantile_transformer(scales)
        instance_pass = self.instance_net(torch.cat([hash, scales], dim=-1))
        
        norms = instance_pass.norm(dim=-1, keepdim=True)
        return instance_pass / (norms + epsilon)

    def get_mlp_3d(self, hash: TensorType, instance_scales: TensorType) -> TensorType:
        """
        Get the relationfield affinity field outputs. Note that this is scale-conditioned.
        This function *does* assume that the hash values are normalized.
        The MLP output is normalized to unit length.
        """
        assert self.quantile_transformer is not None

        # Check that # of rays is the same as # of scales
        assert hash.shape[0] == instance_scales.shape[0]
        epsilon = 1e-5
        if self.use_single_scale:
            instance_pass = self.instance_net(hash)
            return instance_pass / (instance_pass.norm(dim=-1, keepdim=True) + epsilon)

        scales = instance_scales.contiguous().view(-1, 1, 1)

        # Normalize scales before passing to MLP
        scales = self.quantile_transformer(scales)
        instance_pass = self.instance_net(torch.cat([hash, scales], dim=-1).view(-1, hash.shape[-1]+1))
        instance_pass = instance_pass.view(*hash.shape[:2],-1)
        
        norms = instance_pass.norm(dim=-1, keepdim=True)
        return instance_pass / (norms + epsilon)
