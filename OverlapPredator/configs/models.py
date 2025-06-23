from easydict import EasyDict as edict
import torch
import numpy as np
from os.path import join, dirname
from OverlapPredator.lib.utils import load_config
from OverlapPredator.models.architectures import KPFCNN

architectures = dict()
architectures["indoor"] = [
    "simple",
    "resnetb",
    "resnetb_strided",
    "resnetb",
    "resnetb",
    "resnetb_strided",
    "resnetb",
    "resnetb",
    "resnetb_strided",
    "resnetb",
    "resnetb",
    "nearest_upsample",
    "unary",
    "nearest_upsample",
    "unary",
    "nearest_upsample",
    "last_unary",
]
architectures["kitti"] = [
    "simple",
    "resnetb",
    "resnetb_strided",
    "resnetb",
    "resnetb",
    "resnetb_strided",
    "resnetb",
    "resnetb",
    "resnetb_strided",
    "resnetb",
    "resnetb",
    "nearest_upsample",
    "unary",
    "nearest_upsample",
    "unary",
    "nearest_upsample",
    "last_unary",
]
architectures["modelnet"] = [
    "simple",
    "resnetb",
    "resnetb",
    "resnetb_strided",
    "resnetb",
    "resnetb",
    "resnetb_strided",
    "resnetb",
    "resnetb",
    "nearest_upsample",
    "unary",
    "unary",
    "nearest_upsample",
    "unary",
    "last_unary",
]


def load_indoor(device, weights_path):
    config = edict(load_config(join(dirname(__file__), "test", "indoor.yaml")))
    config.device = device
    config.architecture = architectures["indoor"]
    config.model = KPFCNN(config).to(config.device)
    assert config.pretrain != None
    state = torch.load(weights_path)
    config.model.load_state_dict(state["state_dict"])
    return config


def load_kitti(device, weights_path):
    config = edict(load_config(join(dirname(__file__), "test", "kitti.yaml")))
    config.device = device
    config.architecture = architectures["kitti"]
    config.model = KPFCNN(config).to(config.device)
    assert config.pretrain != None
    state = torch.load(weights_path)
    config.model.load_state_dict(state["state_dict"])
    return config


def load_modelnet(device, weights_path):
    config = edict(load_config(join(dirname(__file__), "test", "modelnet.yaml")))
    config.device = device
    config.architecture = architectures["modelnet"]
    config.model = KPFCNN(config).to(config.device)
    assert config.pretrain != None
    state = torch.load(weights_path)
    config.model.load_state_dict(state["state_dict"])
    return config


# https://github.com/prs-eth/OverlapPredator/issues/21
neighborhood_limits = {
    "3dmatch": np.array([38, 36, 36, 38]),
    "kitti": np.array([51, 62, 71, 76]),
}
