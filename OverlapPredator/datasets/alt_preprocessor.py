import numpy as np
import torch
from OverlapPredator.datasets.dataloader import collate_fn_descriptor


def preprocess(src_points, tgt_points, config, neighborhood_limits):
    """
    Convert 2 pointclouds into the format of inputs for KPFCNN model, like ThreeDMatchDemo in scripts/demo.py does

    src_points: np.array([N, 3], dtype=np.float32)
    tgt_points: np.array([M, 3], dtype=np.float32)
    """
    src_feats = np.ones_like(src_points[:, :1]).astype(np.float32)
    tgt_feats = np.ones_like(tgt_points[:, :1]).astype(np.float32)
    # fake the ground truth information
    rot = np.eye(3).astype(np.float32)
    trans = np.ones((3, 1)).astype(np.float32)
    correspondences = torch.ones(1, 2).long()

    config.model.eval()
    inputs = collate_fn_descriptor(
        [(src_points, tgt_points, src_feats, tgt_feats, rot, trans, correspondences, src_points, tgt_points, torch.ones(1))],
        config=config,
        neighborhood_limits=np.array([38, 36, 36, 38]),
    )
    for k, v in inputs.items():
        if type(v) == list:
            inputs[k] = [item.to(config.device) for item in v]
        else:
            inputs[k] = v.to(config.device)
    return inputs
