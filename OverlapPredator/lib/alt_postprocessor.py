import numpy as np


def postprocess(inputs, feats, scores_overlap, scores_saliency, n_points=1000):
    """
    postprocessing of features from KPFCNN to do probabilistic sampling guided by the score, like scripts/demo.py does
    """
    pcd = inputs["points"][0]
    len_src = inputs["stack_lengths"][0][0]
    # c_rot, c_trans = inputs['rot'], inputs['trans']
    # correspondence = inputs['correspondences']

    src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
    # src_raw = copy.deepcopy(src_pcd)
    # tgt_raw = copy.deepcopy(tgt_pcd)
    src_feats, tgt_feats = feats[:len_src], feats[len_src:]
    src_overlap, src_saliency = scores_overlap[:len_src], scores_saliency[:len_src]
    tgt_overlap, tgt_saliency = scores_overlap[len_src:], scores_saliency[len_src:]

    src_scores = src_overlap * src_saliency
    tgt_scores = tgt_overlap * tgt_saliency

    if src_pcd.size(0) > n_points:
        idx = np.arange(src_pcd.size(0))
        probs = (src_scores / src_scores.sum()).detach().cpu().numpy().flatten()
        idx = np.random.choice(idx, size=n_points, replace=False, p=probs)
        src_pcd, src_feats = src_pcd[idx], src_feats[idx]
    if tgt_pcd.size(0) > n_points:
        idx = np.arange(tgt_pcd.size(0))
        probs = (tgt_scores / tgt_scores.sum()).detach().cpu().numpy().flatten()
        idx = np.random.choice(idx, size=n_points, replace=False, p=probs)
        tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

    return src_pcd, tgt_pcd, src_feats.detach(), tgt_feats.detach(), src_overlap.detach(), tgt_overlap.detach()
