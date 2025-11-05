import re
import collections


class Tokenizer:
    """
    A simple Tokenizer class that handles vocabulary building,
    text encoding, decoding, and padding.
    """

    def __init__(self):
        """
        Initializes the tokenizer.
        Sets up special tokens (<pad>, <unk>) and pre-compiles the regex pattern.
        """
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.vocab_size = len(self.word2idx)
        # Pre-compile the regex pattern for efficiency
        self.regex_pattern = re.compile(r"\b\w+\b")

    def normalize_and_split(self, text: str) -> list[str]:
        """
        Normalizes text (to lowercase) and splits it into tokens.

        Args:
            text: The input string to tokenize.

        Returns:
            A list of normalized tokens.
        """
        text = text.lower()
        tokens = self.regex_pattern.findall(text)
        return tokens

    def build_vocab(self, corpus: list[str]):
        """
        Builds the vocabulary from a given corpus.

        Args:
            corpus: A list of sentences (strings) to build the vocab from.
        """
        # 1. Use a generator for memory-efficient iteration over all tokens.
        all_tokens_iter = (
            token for sentence in corpus for token in self.normalize_and_split(sentence)
        )

        # 2. Use collections.Counter to count the frequency of each unique token.
        # token_counts keys are tokens, values are frequencies.
        token_counts = collections.Counter(all_tokens_iter)

        # 3. Add only new tokens (not already in the vocab) to the dictionaries.
        # Note: Sorting token_counts (e.g., by freq) before this loop
        idx = len(self.word2idx)
        for token in token_counts:  # same as token_counts.keys()
            if token not in self.word2idx:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

        self.vocab_size = len(self.word2idx)

    def encode(self, text: str) -> list[int]:
        """
        Converts a text string into a list of token indices.
        Unknown words are mapped to the <unk> token index.

        Args:
            text: The input string.

        Returns:
            A list of token indices.
        """
        tokens = self.normalize_and_split(text)
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        """
        Converts a list of token indices back into a list of token strings.
        Unknown indices are mapped to the <unk> token string.

        Args:
            indices: A list of token indices.

        Returns:
            A list of token strings.
        """
        return [self.idx2word.get(idx, "<unk>") for idx in indices]

    def padding(self, indices: list[int], max_length: int) -> list[int]:
        """
        Pads or truncates a list of indices to a specific max_length.

        Args:
            indices: The list of token indices.
            max_length: The target length.

        Returns:
            A new list of indices with the specified max_length.
        """
        # 1. Truncation: if len(indices) > max_length, cut it down.
        indices = indices[:max_length]

        # 2. Padding: if len(indices) < max_length, fill with <pad> tokens.
        pad_length = max_length - len(indices)
        return indices + [self.word2idx["<pad>"]] * pad_length

    def __call__(self, text: str, max_length: int | None = None) -> list[int]:
        """
        A callable method to encode text and optionally pad/truncate.
        This makes the tokenizer object behave like a function.

        Args:
            text: The input string to tokenize and encode.
            max_length: If provided, the output list will be padded
                        or truncated to this length.

        Returns:
            A list of token indices.
        """
        indices = self.encode(text)
        if max_length is not None:
            indices = self.padding(indices, max_length)
        return indices


if __name__ == "__main__":
    corpus = [
        "The cat sat on the mat.",
        "I love natural language processing.",
        "How are you doing today?",
        "She sells seashells by the seashore.",
        "Artificial intelligence is transforming the world.",
        "Can you believe it's already July?",
        "Let's meet at 5 p.m. in the cafe.",
        "He didn’t know what to say.",
        "The weather is nice and sunny.",
        "Data science combines statistics and programming.",
        "Despite the heavy rain and traffic, she arrived on time with a smile on her face.",
    ]
    tokenizer = Tokenizer()
    tokenizer.build_vocab(corpus)
    encoded = tokenizer(
        "The weather is nice and sunny. This is a test sentence.", max_length=15
    )
    decoded = tokenizer.decode(encoded)
    print("Encoded: ", encoded)
    print("Decoded: ", decoded)
