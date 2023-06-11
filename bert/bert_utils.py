import torch

# from torch.nn import functional as F
# from torch.utils import data

# import re
import random
import os
import requests
import zipfile
import tarfile
import hashlib

from utils import tokenize, count_corpus


DATA_HUB = dict()
DATA_HUB["wikitext-2"] = (
    "https://s3.amazonaws.com/research.metamind.io/wikitext/"
    "wikitext-2-v1.zip",
    "3c914d17d80b1459be871a5039ac23e752a53cbe",
)


def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, "wiki.train.tokens")
    with open(file_name, "r") as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [
        line.strip().lower().split(" . ")
        for line in lines
        if len(line.split(" . ")) >= 2
    ]
    random.shuffle(paragraphs)
    return paragraphs


def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ["<cls>"] + tokens_a + ["<sep>"]
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ["<sep>"]
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs
        )
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(
    tokens, candidate_pred_positions, num_mlm_preds, vocab
):
    # Make a new copy of tokens for the input of a masked language model,
    # where the input may contain replaced '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = "<mask>"
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.randint(0, len(vocab) - 1)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position])
        )
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ["<cls>", "<sep>"]:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab
    )
    pred_positions_and_labels = sorted(
        pred_positions_and_labels, key=lambda x: x[0]
    )
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    (
        all_token_ids,
        all_segments,
        valid_lens,
    ) = (
        [],
        [],
        [],
    )
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (
        token_ids,
        pred_positions,
        mlm_pred_label_ids,
        segments,
        is_next,
    ) in examples:
        all_token_ids.append(
            torch.tensor(
                token_ids + [vocab["<pad>"]] * (max_len - len(token_ids)),
                dtype=torch.long,
            )
        )
        all_segments.append(
            torch.tensor(
                segments + [0] * (max_len - len(segments)), dtype=torch.long
            )
        )
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(
            torch.tensor(
                pred_positions
                + [0] * (max_num_mlm_preds - len(pred_positions)),
                dtype=torch.long,
            )
        )
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor(
                [1.0] * len(mlm_pred_label_ids)
                + [0.0] * (max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32,
            )
        )
        all_mlm_labels.append(
            torch.tensor(
                mlm_pred_label_ids
                + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)),
                dtype=torch.long,
            )
        )
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (
        all_token_ids,
        all_segments,
        valid_lens,
        all_pred_positions,
        all_mlm_weights,
        all_mlm_labels,
        nsp_labels,
    )


class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self.token_freqs = sorted(
            counter.items(), key=lambda x: x[1], reverse=True
        )
        # The index for the unknown token is 0
        self.unk, uniq_tokens = 0, ["<unk>"] + reserved_tokens
        uniq_tokens += [
            token
            for token, freq in self.token_freqs
            if freq >= min_freq and token not in uniq_tokens
        ]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [
            tokenize(paragraph, token="word") for paragraph in paragraphs
        ]
        sentences = [
            sentence for paragraph in paragraphs for sentence in paragraph
        ]
        self.vocab = Vocab(
            sentences,
            min_freq=5,
            reserved_tokens=["<pad>", "<mask>", "<cls>", "<sep>"],
        )
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(
                _get_nsp_data_from_paragraph(
                    paragraph, paragraphs, self.vocab, max_len
                )
            )
        # Get data for the masked language model task
        examples = [
            (
                _get_mlm_data_from_tokens(tokens, self.vocab)
                + (segments, is_next)
            )
            for tokens, segments, is_next in examples
        ]
        # Pad inputs
        (
            self.all_token_ids,
            self.all_segments,
            self.valid_lens,
            self.all_pred_positions,
            self.all_mlm_weights,
            self.all_mlm_labels,
            self.nsp_labels,
        ) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (
            self.all_token_ids[idx],
            self.all_segments[idx],
            self.valid_lens[idx],
            self.all_pred_positions[idx],
            self.all_mlm_weights[idx],
            self.all_mlm_labels[idx],
            self.nsp_labels[idx],
        )

    def __len__(self):
        return len(self.all_token_ids)


# Required functions for downloading data


def download(name, cache_dir=os.path.join("..", "data")):
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                _data = f.read(1048576)
                if not _data:
                    break
                sha1.update(_data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f"Downloading {fname} from {url}...")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, "wb") as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        fp = zipfile.ZipFile(fname, "r")
    elif ext in (".tar", ".gz"):
        fp = tarfile.open(fname, "r")
    else:
        assert False, "Only zip/tar files can be extracted."
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def load_data_wiki(batch_size, max_len):
    num_workers = 4
    data_dir = download_extract("wikitext-2", "wikitext-2")
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(
        train_set, batch_size, shuffle=True, num_workers=num_workers
    )
    return train_iter, train_set.vocab
