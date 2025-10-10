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
EXT_split_pattern = None
EXT_special_tokens_len = 0

def init_worker(
    PAT: str, 
    split_pattern: str,
    input_path: str,
    special_tokens_len: int
):
    global EXT_PAT, EXT_split_pattern, EXT_input_path, EXT_special_tokens_len

    EXT_PAT = regex.compile(PAT)
    EXT_input_path = input_path
    EXT_split_pattern = regex.compile(split_pattern) if split_pattern else None
    EXT_special_tokens_len = special_tokens_len

def byte2id(byte: int):
    return EXT_special_tokens_len + byte

def build_corpus_worker(args):
    start, end = args
    corpus = []

    with open(EXT_input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    if EXT_split_pattern:
        docs = EXT_split_pattern.split(chunk)
    else:
        docs = [chunk]

    # Run pre-tokenization on the chunk
    for doc in docs:
        for m in EXT_PAT.finditer(doc):
            b = m.group(0).encode("utf-8") # find exact substr & convert to utf-8 range 0 ~ 255
            id_seq = [byte2id(byte) for byte in b]
            
            if id_seq:
                corpus.append(id_seq)

    return corpus

def apply_merge(
    corpus: list[list[int]],
    id_pair_cnts: Counter,
    new_id: int,    # AB
    id_a: int,      # A
    id_b: int,      # B
):
    # When you merge tokens (A, B) → AB in a sequence like [..., X, A, B, Y, ...]:
    # Remove these pairs: (X, A), (A, B), (B, Y)
    # Add these new pairs: (X, AB), (AB, Y)
    for i, id_seq in enumerate(corpus):
        if id_a not in id_seq:
            continue
        
        j, n = 0, len(id_seq)
        new_id_seq = []

        while j < n:
            if j+1 < n and id_seq[j] == id_a and id_seq[j+1] == id_b:
                prev_id : int = None
                next_id : int = None

                if len(new_id_seq) > 0: # if prev_id exists, delete (X, A)
                    prev_id = new_id_seq[-1]
                    id_pair_cnts[(prev_id, id_a)] -= 1
                    if id_pair_cnts[(prev_id, id_a)] == 0:
                        del id_pair_cnts[(prev_id, id_a)]
                
                if j+2 < len(id_seq): # if next_id exists, skips B & get next_id, delete (B, Y)
                    next_id = id_seq[j+2]
                    id_pair_cnts[(id_b, next_id)] -= 1
                    if id_pair_cnts[(id_b, next_id)] == 0:
                        del id_pair_cnts[(id_b, next_id)]
                
                # delete (A, B)
                id_pair_cnts[(id_a, id_b)] -= 1
                if id_pair_cnts[(id_a, id_b)] == 0:
                    del id_pair_cnts[(id_a, id_b)]
                
                # add (X, AB) & (AB, Y)
                if prev_id is not None:
                    id_pair_cnts[(prev_id, new_id)] += 1
                if next_id is not None:
                    id_pair_cnts[(new_id, next_id)] += 1

                new_id_seq.append(new_id)
                j += 2
            else:
                new_id_seq.append(id_seq[j])
                j += 1
        
        corpus[i] = new_id_seq

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
    split_pattern = "|".join(re.escape(token) for token in special_tokens if token)
    if not split_pattern:
        split_pattern = None
    
    num_processes = max(4, os.cpu_count() or 1)

    with open(input_path, "rb") as f:
        # if boundaries = [b0, b1, b2, b3], then boundaries[:-1] = [b0, b1, b2], boundaries[1:] = [b1, b2, b3]
        # start, end = (b0 b1), (b1 b2), (b2 b3) consecutive pairs
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        num_jobs = list(zip(boundaries[:-1], boundaries[1:]))

        with Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(PAT, split_pattern, input_path, len(special_tokens)),
        ) as pool:
            # unordered: results yielded in completion order, not original input order
            # imap(...): results yielded in input order
            for seqs in pool.imap_unordered(build_corpus_worker, num_jobs, chunksize=1):
                corpus.extend(seqs)
    
    id_pair_cnts = Counter()
    for seq in corpus:
        for i in range(len(seq) - 1):
            id_pair_cnts[(seq[i], seq[i+1])] += 1

    # learn merge until vocab_size, since merging adds 1 new token into vocab
    while len(id_to_bytes) < vocab_size:
        if not id_pair_cnts:
            break

        # Find the highest freq pair, with deterministic tie-breaking
        # When counts are tied, select lexicographically largest bytes (matching reference)
        max_count = max(id_pair_cnts.values())
        id_a, id_b = max(
            (pair for pair, count in id_pair_cnts.items() if count == max_count),
            key=lambda p: (id_to_bytes[p[0]], id_to_bytes[p[1]])
        )
        new_bytes = id_to_bytes[id_a] + id_to_bytes[id_b]
        if new_bytes in bytes_to_id:
            break

        new_id = len(id_to_bytes)
        id_to_bytes[new_id] = new_bytes
        bytes_to_id[new_bytes] = new_id
        merges.append((id_to_bytes[id_a], id_to_bytes[id_b]))

        apply_merge(corpus, id_pair_cnts, new_id, id_a, id_b)

    return id_to_bytes, merges