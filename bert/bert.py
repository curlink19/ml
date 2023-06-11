from torch import nn
from encoder import BERTEncoder
from mask_lm import MaskLM
from next_sent_pred import NextSentencePred


class BERTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        max_len=1000,
        key_size=768,
        query_size=768,
        value_size=768,
        hid_in_features=768,
        mlm_in_features=768,
        nsp_in_features=768,
    ):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(
            vocab_size,
            num_hiddens,
            norm_shape,
            ffn_num_input,
            ffn_num_hiddens,
            num_heads,
            num_layers,
            dropout,
            max_len=max_len,
            key_size=key_size,
            query_size=query_size,
            value_size=value_size,
        )
        self.hidden = nn.Sequential(
            nn.Linear(hid_in_features, num_hiddens), nn.Tanh()
        )
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
