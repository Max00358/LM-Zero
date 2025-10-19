import os, re
import json
import numpy as np

from .pretokenization_example import find_chunk_boundaries
from tests.common import gpt2_bytes_to_unicode

from pathlib import Path
from collections import Counter
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Iterator
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# uv run python -m cProfile -o profile.stats -m pytest
# uv run python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(30)"

BYTE_DECODER = {v: k for k, v in gpt2_bytes_to_unicode().items()}

class tokenizer:
    def __init__(
            self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] | None = None
        ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    # Constructs and return a Tokenizer from a serialized vocabulary and list of merges
    @classmethod
    def from_files(
        # class, require @classmethod for Python to recognize it, else it will be treated as a regular arg
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        vocab_filepath = Path(vocab_filepath)
        merges_filepath = Path(merges_filepath)

        if not vocab_filepath.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_filepath}")
        if not merges_filepath.exists():
            raise FileNotFoundError(f"Merges file not found: {merges_filepath}")

        vocab = cls._load_vocab(vocab_filepath)     # id2bytes
        merges = cls._load_merges(merges_filepath)  # (bytes & bytes)
        return cls(
            vocab, 
            merges,
            special_tokens
        )

    @staticmethod # no self or cls
    def _load_vocab(vocab_filepath: Path):
        id_to_bytes: dict[int, bytes] = {}

        text = vocab_filepath.read_text(encoding="utf-8", errors="ignore")
        data = json.loads(text)

        for token_str, token_id in data.items():
            token_id = int(token_id)
            token_bytes = bytes([BYTE_DECODER[byte] for byte in token_str])
            id_to_bytes[token_id] = token_bytes

        return id_to_bytes
    
    @staticmethod
    def _load_merges(merges_filepath: Path):
        merges: list[tuple[bytes, bytes]] = []

        with merges_filepath.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Bad merges line (need 2 columns): {line}")

                merges.append(
                    (
                        bytes([BYTE_DECODER[byte] for byte in parts[0]]),
                        bytes([BYTE_DECODER[byte] for byte in parts[1]])
                    )
                )
        
        return merges

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass


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

    EXT_PAT = re.compile(PAT)
    EXT_input_path = input_path
    EXT_split_pattern = re.compile(split_pattern) if split_pattern else None
    EXT_special_tokens_len = special_tokens_len

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
            
            arr = np.frombuffer(b, dtype=np.uint8)
            id_seq = (EXT_special_tokens_len + arr).tolist()
            # id_seq = [EXT_special_tokens_len + byte for byte in b] # convert byte to id
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
    
    # Pattern converted from regex to re:
        # \p{L} (letters) → [^\W\d_] (word chars excluding digits and underscore)
        # \p{N} (numbers) → \d
        # [^\s\p{L}\p{N}] (not space/letter/number) → [^\s\w] (not space/word)
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
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
        id_pair_cnts.update(zip(seq, seq[1:]))

    # learn merge until vocab_size, since merging adds 1 new token into vocab
    num_merges = vocab_size - len(id_to_bytes)

    with tqdm(total=num_merges, desc="Merging tokens...", unit="merge") as pbar:
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
            pbar.update(1)

    return id_to_bytes, merges

def save_bpe(
    id_to_bytes: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_path: str
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    byte_encoder = gpt2_bytes_to_unicode()    
    vocab = {}
    for token_id, token_bytes in id_to_bytes.items():
        # byte_encoder converts bytes to utf-8 strings, 20746865 -> "the"
        token_str = "".join(byte_encoder[byte] for byte in token_bytes)
        vocab[token_str] = token_id
    
    with open(output_path / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    with open(output_path / "merges.txt", "w", encoding="utf-8") as f:
        for x, y in merges:
            x_str = "".join(byte_encoder[byte] for byte in x)
            y_str = "".join(byte_encoder[byte] for byte in y)
            f.write(f"{x_str} {y_str}\n")
    
    print(f"✓ Saved vocab ({len(vocab)} tokens) and merges ({len(merges)}) to {output_path}")

def vocab_info(
    id_to_bytes: dict[int, bytes],
):
    # longest token by byte len
    longest_id, longest_bytes = max(id_to_bytes.items(), key=lambda x: len(x[1]))

    try: 
        longest_bytes_utf8 = longest_bytes.decode("utf-8")
    except UnicodeDecodeError:
        longest_bytes_utf8 = None

    print (
        f"\n\t========== Vocab Info ==========\n",
        f"\tLongest ID: {longest_id}\n",
        f"\tLongest Bytes Length: {len(longest_bytes)}\n",
        f"\tLongest Bytes UTF-8: {longest_bytes_utf8 if longest_bytes_utf8 is not None else '<non-utf8>'}\n"
    )