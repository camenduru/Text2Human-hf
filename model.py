from __future__ import annotations

import pathlib
import sys
import zipfile

import huggingface_hub
import numpy as np
import PIL.Image
import torch

sys.path.insert(0, 'Text2Human')

from models.sample_model import SampleFromPoseModel
from utils.language_utils import (generate_shape_attributes,
                                  generate_texture_attributes)
from utils.options import dict_to_nonedict, parse
from utils.util import set_random_seed

COLOR_LIST = [
    (0, 0, 0),
    (255, 250, 250),
    (220, 220, 220),
    (250, 235, 215),
    (255, 250, 205),
    (211, 211, 211),
    (70, 130, 180),
    (127, 255, 212),
    (0, 100, 0),
    (50, 205, 50),
    (255, 255, 0),
    (245, 222, 179),
    (255, 140, 0),
    (255, 0, 0),
    (16, 78, 139),
    (144, 238, 144),
    (50, 205, 174),
    (50, 155, 250),
    (160, 140, 88),
    (213, 140, 88),
    (90, 140, 90),
    (185, 210, 205),
    (130, 165, 180),
    (225, 141, 151),
]


class Model:
    def __init__(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = self._load_config()
        self.config['device'] = device.type
        self._download_models()
        self.model = SampleFromPoseModel(self.config)
        self.model.batch_size = 1

    def _load_config(self) -> dict:
        path = 'Text2Human/configs/sample_from_pose.yml'
        config = parse(path, is_train=False)
        config = dict_to_nonedict(config)
        return config

    def _download_models(self) -> None:
        model_dir = pathlib.Path('pretrained_models')
        if model_dir.exists():
            return
        path = huggingface_hub.hf_hub_download('yumingj/Text2Human_SSHQ',
                                               'pretrained_models.zip')
        model_dir.mkdir()
        with zipfile.ZipFile(path) as f:
            f.extractall(model_dir)

    @staticmethod
    def preprocess_pose_image(image: PIL.Image.Image) -> torch.Tensor:
        image = np.array(
            image.resize(
                size=(256, 512),
                resample=PIL.Image.Resampling.LANCZOS))[:, :, 2:].transpose(
                    2, 0, 1).astype(np.float32)
        image = image / 12. - 1
        data = torch.from_numpy(image).unsqueeze(1)
        return data

    @staticmethod
    def process_mask(mask: np.ndarray) -> np.ndarray:
        if mask.shape != (512, 256, 3):
            return None
        seg_map = np.full(mask.shape[:-1], -1)
        for index, color in enumerate(COLOR_LIST):
            seg_map[np.sum(mask == color, axis=2) == 3] = index
        if not (seg_map != -1).all():
            return None
        return seg_map

    @staticmethod
    def postprocess(result: torch.Tensor) -> np.ndarray:
        result = result.permute(0, 2, 3, 1)
        result = result.detach().cpu().numpy()
        result = result * 255
        result = np.asarray(result[0, :, :, :], dtype=np.uint8)
        return result

    def process_pose_image(self, pose_image: PIL.Image.Image) -> torch.Tensor:
        if pose_image is None:
            return
        data = self.preprocess_pose_image(pose_image)
        self.model.feed_pose_data(data)
        return data

    def generate_label_image(self, pose_data: torch.Tensor,
                             shape_text: str) -> np.ndarray:
        if pose_data is None:
            return
        self.model.feed_pose_data(pose_data)
        shape_attributes = generate_shape_attributes(shape_text)
        shape_attributes = torch.LongTensor(shape_attributes).unsqueeze(0)
        self.model.feed_shape_attributes(shape_attributes)
        self.model.generate_parsing_map()
        self.model.generate_quantized_segm()
        colored_segm = self.model.palette_result(self.model.segm[0].cpu())
        return colored_segm

    def generate_human(self, label_image: np.ndarray, texture_text: str,
                       sample_steps: int, seed: int) -> np.ndarray:
        if label_image is None:
            return
        mask = label_image.copy()
        seg_map = self.process_mask(mask)
        if seg_map is None:
            return
        self.model.segm = torch.from_numpy(seg_map).unsqueeze(0).unsqueeze(
            0).to(self.model.device)
        self.model.generate_quantized_segm()

        set_random_seed(seed)

        texture_attributes = generate_texture_attributes(texture_text)
        texture_attributes = torch.LongTensor(texture_attributes)
        self.model.feed_texture_attributes(texture_attributes)
        self.model.generate_texture_map()

        self.model.sample_steps = sample_steps
        out = self.model.sample_and_refine()
        res = self.postprocess(out)
        return res
