# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper for performing DINOv2 inference."""

import cv2
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "third_party"))
from dinov2 import dino

from . import utils
import torch


class DINOExtract:
    """Class to initialize DINO model and extract features from an image."""

    def __init__(self, cpt_path: str, feature_layer: int = 1):
        self.feature_layer = feature_layer
        self.model = dino.vit_base()
        state_dict_raw = torch.load(cpt_path, map_location="cpu")

        # state_dict = {}
        # for k, v in state_dict_raw.items():
        #   state_dict[k.replace('blocks', 'blocks.0')] = v

        self.model.load_state_dict(state_dict_raw)
        self.model.eval()

        self.image_size_max = 630

        self.h_down_rate = self.model.patch_embed.patch_size[0]
        self.w_down_rate = self.model.patch_embed.patch_size[1]

    def __call__(self, image) -> np.ndarray:
        return self.forward(image)

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Feeds image through DINO ViT model to extract features.

        Args:
          image: (H, W, 3) numpy array, decoded image bytes, value range [0, 255].

        Returns:
          features: (H // 14, W // 14, C) numpy array image features.
        """
        image = self._resize_input_image(image)
        image_processed = self._process_image(image)
        image_processed = image_processed.unsqueeze(0).float()
        features = self.extract_feature(image_processed)
        features = features.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return features

    def _resize_input_image(
        self, image: np.ndarray, interpolation=cv2.INTER_LINEAR
    ):
        """Resizes image such that both dimensions are divisble by down_rate."""
        h_image, w_image = image.shape[:2]
        h_larger_flag = h_image > w_image
        large_side_image = max(h_image, w_image)

        # resize the image with the largest side length smaller than a threshold
        # to accelerate ViT backbone inference (which has quadratic complexity).
        if large_side_image > self.image_size_max:
            if h_larger_flag:
                h_image_target = self.image_size_max
                w_image_target = int(self.image_size_max * w_image / h_image)
            else:
                w_image_target = self.image_size_max
                h_image_target = int(self.image_size_max * h_image / w_image)
        else:
            h_image_target = h_image
            w_image_target = w_image

        h, w = (
            h_image_target // self.h_down_rate,
            w_image_target // self.w_down_rate,
        )
        h_resize, w_resize = h * self.h_down_rate, w * self.w_down_rate
        image = cv2.resize(
            image, (w_resize, h_resize), interpolation=interpolation
        )
        return image

    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        """Turn image into pytorch tensor and normalize it."""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image_processed = image / 255.0
        image_processed = (image_processed - mean) / std
        image_processed = torch.from_numpy(image_processed).permute(2, 0, 1)
        return image_processed

    def extract_feature(self, image):
        """Extracts features from image.

        Args:
          image: (B, 3, H, W) torch tensor, normalized with ImageNet mean/std.

        Returns:
          features: (B, C, H//14, W//14) torch tensor image features.
        """
        b, _, h_origin, w_origin = image.shape
        out = self.model.get_intermediate_layers(image, n=self.feature_layer)[0]
        h = int(h_origin / self.h_down_rate)
        w = int(w_origin / self.w_down_rate)
        dim = out.shape[-1]
        out = out.reshape(b, h, w, dim).permute(0, 3, 1, 2).detach()
        return out


def _preprocess_shape(
    h_image, w_image, image_size_max=630, h_down_rate=14, w_down_rate=14
):
    h_image = h_image.squeeze()
    w_image = w_image.squeeze()

    h_larger_flag = h_image > w_image
    large_side_image = max(h_image, w_image)

    def resize_h_larger():
        h_image_target = image_size_max
        w_image_target = int(image_size_max * w_image / h_image)
        return h_image_target, w_image_target

    def resize_w_larger_or_equal():
        w_image_target = image_size_max
        h_image_target = int(image_size_max * h_image / w_image)
        return h_image_target, w_image_target

    def keep_original():
        return h_image, w_image

    if large_side_image > image_size_max:
        if h_larger_flag:
            h_image_target, w_image_target = resize_h_larger()
        else:
            h_image_target, w_image_target = resize_w_larger_or_equal()
    else:
        h_image_target, w_image_target = keep_original()

    h = h_image_target // h_down_rate
    w = w_image_target // w_down_rate
    h_resize = torch.tensor(h * h_down_rate)
    w_resize = torch.tensor(w * w_down_rate)

    h_resize = h_resize.unsqueeze(0)
    w_resize = w_resize.unsqueeze(0)

    return h_resize, w_resize


def get_dino_descriptors(dino_features, keypoints, height, width, feature_dim):
    """Get DINO descriptors using Superpoint keypoints.

    Args:
        dino_features: DINO features in 1-D.
        keypoints: Superpoint keypoint locations, in format (x, y), in pixels, shape
        (N, 2).
        height: image height, type torch int32.
        width: image width, type torch int32.
        feature_dim: DINO feature channel size, type torch int32.

    Returns:
        Interpolated DINO descriptors.
    """
    height_1d = height.reshape([1])
    width_1d = width.reshape([1])

    height_1d_resized, width_1d_resized = _preprocess_shape(
        height_1d, width_1d, image_size_max=630, h_down_rate=14, w_down_rate=14
    )

    height_feat = height_1d_resized // 14
    width_feat = width_1d_resized // 14
    feature_dim_1d = torch.tensor(feature_dim).reshape([1])

    dino_features = dino_features.reshape(
        height_feat, width_feat, feature_dim_1d
    )

    img_size = torch.cat([width_1d, height_1d], dim=0).float()
    feature_size = torch.cat([width_feat, height_feat], dim=0).float()
    keypoints_feature = (
        keypoints[0] / img_size.unsqueeze(0) * feature_size.unsqueeze(0)
    )

    dino_descriptors = []
    for kp in keypoints_feature:
        dino_descriptors.append(
            utils.lookup_descriptor_bilinear(kp.numpy(), dino_features)
        )
    dino_descriptors = torch.tensor(
        np.array(dino_descriptors), dtype=torch.float32
    )
    return dino_descriptors
