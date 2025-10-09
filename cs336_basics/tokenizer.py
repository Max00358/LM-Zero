import os, re
import regex
from .pretokenization_example import find_chunk_boundaries

from collections import Counter
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
from multiprocessing import Pool, cpu_count

# uv run python -m cProfile -o profile.stats -m pytest
# uv run python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(30)"
def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError

EXT_PAT = None
EXT_input_path = None
EXT_special_tokens_len = 0

def init_worker(
    PAT: str, 
    input_path: str,
    special_tokens_len: int
):
    global EXT_PAT, EXT_input_path, EXT_special_tokens_len

    EXT_PAT = regex.compile(PAT)
    EXT_input_path = input_path
    EXT_special_tokens_len = special_tokens_len

def byte2id(byte: int):
    return EXT_special_tokens_len + byte

def build_corpus_worker(args):
    start, end = args
    corpus = []

    with open(EXT_input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # Run pre-tokenization on the chunk
    for m in EXT_PAT.finditer(chunk):
        b = m.group(0).encode("utf-8") # find exact substr & convert to utf-8 range 0 ~ 255
        id_seq = [byte2id(byte) for byte in b]
        
        if id_seq:
            corpus.append(id_seq)

    return corpus

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    id_to_bytes: dict[int, bytes] = {}  # vocab: map int to token bytes, tokenizer's LUT
    bytes_to_id: dict[bytes, int] = {}
    merges: list[tuple[bytes, bytes]] = []
    corpus: list[list[int]] = []

    curr_id = 0
    for token in special_tokens:
        b = token.encode('utf-8')
        bytes_to_id[b] = curr_id
        id_to_bytes[curr_id] = b
        curr_id += 1
    for byte in range(256):
        b = bytes([byte]) # bytes(byte) returns 'byte' number of zeroes in bytes
        bytes_to_id[b] = curr_id
        id_to_bytes[curr_id] = b
        curr_id += 1
    
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    num_processes = max(4, os.cpu_count() or 1)

    with open(input_path, "rb") as f:
        # if boundaries = [b0, b1, b2, b3], then boundaries[:-1] = [b0, b1, b2], boundaries[1:] = [b1, b2, b3]
        # start, end = (b0 b1), (b1 b2), (b2 b3) consecutive pairs
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        num_jobs = list(zip(boundaries[:-1], boundaries[1:]))

        with Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(PAT, input_path, len(special_tokens)),
        ) as pool:
            # unordered: results yielded in completion order, not original input order
            # imap(...): results yielded in input order
            for seqs in pool.imap_unordered(build_corpus_worker, num_jobs, chunksize=1):
                corpus.extend(seqs)
    
    # learn merge until vocab_size since merging adds 1 new token into vocab
    while len(id_to_bytes) < vocab_size:
        id_pair_cnts = Counter()
        u = id_pair_cnts.update
        for seq in corpus:
            u(zip(seq, seq[1:]))
        if not id_pair_cnts:
            break

        # Find the most common pair, with deterministic tie-breaking
        # When counts are tied, select lexicographically largest bytes (matching reference)
        max_count = max(id_pair_cnts.values())
        best_pair = max(
            (pair for pair, count in id_pair_cnts.items() if count == max_count),
            key=lambda p: (id_to_bytes[p[0]], id_to_bytes[p[1]])
        )
        id_a, id_b = best_pair
        new_bytes = id_to_bytes[id_a] + id_to_bytes[id_b]
        if new_bytes in bytes_to_id:
            break

        new_id = len(id_to_bytes)
        id_to_bytes[new_id] = new_bytes
        bytes_to_id[new_bytes] = new_id
        merges.append((id_to_bytes[id_a], id_to_bytes[id_b]))

        # apply merge
        for i, id_seq in enumerate(corpus):
            j, n = 0, len(id_seq)
            new_id_seq = []

            while j < n:
                if j+1 < n and id_seq[j] == id_a and id_seq[j+1] == id_b:
                    new_id_seq.append(new_id)
                    j += 2
                else:
                    new_id_seq.append(id_seq[j])
                    j += 1
            corpus[i] = new_id_seq

    return id_to_bytes, merges