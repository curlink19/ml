from bert_utils import Vocab, DATA_HUB, download_extract
import json
import os
import torch
from bert import BERTModel

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATA_HUB["bert.base"] = (
    DATA_URL + "bert.base.torch.zip",
    "225d66f04cae318b841a13d32af3acc165f253ac",
)
DATA_HUB["bert.small"] = (
    DATA_URL + "bert.small.torch.zip",
    "c72329e68a732bef0452e4b96a1c341c8910f81f",
)


def load_pretrained_model(
    pretrained_model,
    num_hiddens,
    ffn_num_hiddens,
    num_heads,
    num_layers,
    dropout,
    max_len,
    devices,
):
    data_dir = download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, "vocab.json")))
    vocab.token_to_idx = {
        token: idx for idx, token in enumerate(vocab.idx_to_token)
    }
    bert = BERTModel(
        len(vocab),
        num_hiddens,
        norm_shape=[256],
        ffn_num_input=256,
        ffn_num_hiddens=ffn_num_hiddens,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
        max_len=max_len,
        key_size=256,
        query_size=256,
        value_size=256,
        hid_in_features=256,
        mlm_in_features=256,
        nsp_in_features=256,
    )
    # Load pretrained BERT parameters
    bert.load_state_dict(
        torch.load(os.path.join(data_dir, "pretrained.params"))
    )
    return bert, vocab
