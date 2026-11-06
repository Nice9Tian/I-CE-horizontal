"""ice_h: overlapping-group I-CE on Steam reviews (K content anchors per game).

Modules
    data      fixtures (games, folds, queries, tags), the training sampler over the
              in-RAM review pool, the paper's evaluation draw rule and its cache
    model     the latent array with a Sinkhorn membership matrix and Qwen-space anchors
    losses    the I-CE term with decayed negatives, the view-invariance term, monitors
    evaluate  encoding packs and queries, set-to-set retrieval scores, the tag probes
"""
