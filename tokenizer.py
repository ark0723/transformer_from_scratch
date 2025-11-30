import re
import collections
import numpy as np


class Tokenizer:
    """
    A simple Tokenizer class that handles vocabulary building,
    text encoding, decoding, and padding.
    """

    def __init__(self):
        """
        Initializes the tokenizer.
        Sets up special tokens (<pad>, <unk>, <sos>, <eos>) and pre-compiles the regex pattern.
        """
        self.specials = ["<pad>", "<unk>", "<sos>", "<eos>"]
        self.word2idx = {word: i for i, word in enumerate(self.specials)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}

        # IDs are pre-saved for readability and speed improvement
        self.pad_token_id = self.word2idx["<pad>"]
        self.unk_token_id = self.word2idx["<unk>"]
        self.sos_token_id = self.word2idx["<sos>"]
        self.eos_token_id = self.word2idx["<eos>"]

        # Pre-compile the regex pattern for efficiency
        self.regex_pattern = re.compile(r"\b\w+\b")
        self.vocab_size = len(self.word2idx)
        self.vocab_array = None

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
        self.vocab_array = np.array(
            [self.idx2word[i] for i in range(len(self.idx2word))]
        )

    def encode(
        self, text: str, add_sos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """
        Converts a text string into a list of token indices.
        Unknown words are mapped to the <unk> token index.

        Args:
            text: The input string.

        Returns:
            A list of token indices.
        """
        tokens = self.normalize_and_split(text)
        ids = [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]

        if add_sos:
            ids = [self.sos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(
        self,
        indices: list[int] | list[list[int]] | np.ndarray,
        skip_special_tokens: bool = False,
    ) -> np.ndarray:
        """
        Converts token indices back into token strings using NumPy indexing.
        This supports both single sequences and batches.

        Args:
            indices: A list of token indices (or a list of lists/NumPy array).
            skip_special_tokens: If True, removes special tokens from the output.

        Returns:
            A NumPy array of token strings.
        """
        # 1. Convert list into numpy array if it isn't one already
        indices_arr = (
            np.array(indices) if not isinstance(indices, np.ndarray) else indices
        )
        decoded = self.vocab_array[indices_arr]

        if skip_special_tokens:
            # 3. Vectorized Filtering using np.isin
            # self.specials에 포함되지 않은(invert=True) 요소만 True인 마스크 생성
            mask = np.isin(decoded, self.specials, invert=True)

            # 4. 마스크 적용하여 반환
            # 주의: 입력이 2D(배치)였어도, 필터링 후에는 길이가 제각각이므로
            # 1D array로 평탄화(flatten) 되어 반환
            return decoded[mask]

        return decoded

    def __call__(
        self,
        text: str | list[str],
        max_length: int | None = None,
        add_sos: bool = False,
        add_eos: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Encodes text and applies padding/truncation using NumPy.
        This method supports both single strings and lists of strings (batch mode).

        Args:
            text: The input string or list of strings to tokenize.
            max_length: The target length for padding/truncation.

        Returns:
            A dictionary containing:
            - 'input_ids': (batch_size, max_length)
            - 'attention_mask': (batch_size, max_length) -> 1 for real token, 0 for <pad>
        """
        # 1. Input normalization: if text is a string, convert it to a list to unify processing
        if isinstance(text, str):
            text = [text]

        # 2. Encoding: Convert text to integer sequences using list comprehension
        batch_indices = [
            self.encode(sentence, add_sos=add_sos, add_eos=add_eos) for sentence in text
        ]
        # 3. Padding and Batching: Create an optimized NumPy array
        batch_size = len(batch_indices)

        # Create a NumPy array filled with padding tokens (0)
        input_ids = np.full((batch_size, max_length), self.pad_token_id, dtype=np.int32)

        # initialize attention mask filled with 0(pad)
        attention_mask = np.zeros((batch_size, max_length), dtype=np.int32)

        for i, indices in enumerate(batch_indices):
            # compute the truncated length
            length = min(len(indices), max_length)
            # Copy the truncated indices to the padded array
            input_ids[i, :length] = indices[:length]
            # 1 for real token, 0 for <pad>
            attention_mask[i, :length] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}


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

    # 1. single encoding test
    single_output = tokenizer("Deep learning is fun.", max_length=10)
    print(
        f"Single Input IDs: {single_output['input_ids']} / Single Attention Mask: {single_output['attention_mask']}"
    )

    # 2. batch encoding test
    batch_input = ["The cat sat on the mat.", "I love natural language processing."]
    batch_output = tokenizer(batch_input, max_length=6)
    print(
        f"Batch Input IDs: {batch_output['input_ids']} / Batch Attention Mask: {batch_output['attention_mask']}"
    )
    print(f"Batch Decoded: {tokenizer.decode(batch_output['input_ids'])}")
