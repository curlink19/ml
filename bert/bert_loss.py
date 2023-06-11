def _get_batch_loss_bert(
    net,
    loss,
    vocab_size,
    tokens_X,
    segments_X,
    valid_lens_x,
    pred_positions_X,
    mlm_weights_X,
    mlm_Y,
    nsp_y,
):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(
        tokens_X, segments_X, valid_lens_x.reshape(-1), pred_positions_X
    )
    # Compute masked language model loss
    mlm_l = loss(
        mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)
    ) * mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l__ = mlm_l + nsp_l
    return mlm_l, nsp_l, l__
