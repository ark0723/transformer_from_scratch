import re
import collections
import numpy as np


class BPETokenizer:
    def __init__(
        self,
        min_frequency: int = 1,
        special_tokens: list[str] = ["<pad>", "<unk>", "<sos>", "<eos>"],
    ):
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens

        # initialize vocabulary
        self.word2idx = {word: i for i, word in enumerate(self.special_tokens)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}

        # BPE merge dictionary (pair -> new_token)
        self.vocab_size = len(self.word2idx)
        self.merges = {}  # merge rules for encoding : pair -> new_token
        self.ranks = {}  # pair -> rank

    def initialize(self, corpus: list[str]) -> dict[tuple[str, ...], int]:
        """
        코퍼스를 (단어 튜플, 빈도수) 형태의 딕셔너리로 변환
        Example: {'l o w </w>': 5, 'l o w e r </w>': 2}
        """
        # 1. 모든 단어를 분리하여 카운트
        word_counts = collections.Counter()
        for sentence in corpus:
            # 2. 각 문장을 space 기준으로 단어로 분리하고 각 단어를 카운트
            words = sentence.strip().split()
            word_counts.update(words)

        # 2. 문자 단위 분리 + </w> (space) : tuple로 변환하여 딕셔너리 키로 사용
        vocab = {
            tuple(list(word.lower()) + ["<w/>"]): freq
            for word, freq in word_counts.items()
        }
        return vocab

    def get_adjacent_pairs(
        self, vocab: dict[tuple[str, ...], int]
    ) -> collections.Counter:
        """
        현재 어휘 집합에서 인접한 쌍의 빈도를 계산
        """
        pairs = collections.Counter()
        for word_tuple, freq in vocab.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    def merge_pair(
        self, best_pair: tuple[str, str], vocab: dict[tuple[str, ...], int]
    ) -> dict[tuple[str, ...], int]:
        """
        Merge a pair of tokens in the vocabulary and return the updated vocabulary and the new token.
        Example:
        vocab = {'l o w </w>': 5, 'l o w e r </w>': 2}
        best_pair = ('l', 'o')
        new_vocab = {'lo w </w>': 5, 'lo w e r </w>': 2}
        new_token = 'lo'
        """
        first, second = best_pair
        new_token = first + second
        new_vocab = {}

        # vocab의 key(단어 튜플)를 순회하며 replace 수행
        for word_tuple, freq in vocab.items():
            new_word_tuple = []
            i = 0
            while i < len(word_tuple):
                # 인접한 두 문자가 병합 대상과 일치하는지 확인
                if (
                    i < len(word_tuple) - 1
                    and word_tuple[i] == first
                    and word_tuple[i + 1] == second
                ):
                    new_word_tuple.append(new_token)
                    i += 2
                else:
                    new_word_tuple.append(word_tuple[i])
                    i += 1

            # 갱신된 튜플을 키로 사용하여 빈도수 유지
            new_vocab[tuple(new_word_tuple)] = freq

        return new_vocab, new_token

    def train(self, corpus: list[str]) -> tuple[dict[str, int], dict[int, str]]:
        """
        Train the BPE tokenizer on a given corpus and return the word2idx and idx2word dictionaries.
        """
        # 1. 초기화 (단어 튜플, 빈도수) 형태의 딕셔너리로 변환
        vocab = self.initialize(corpus)
        rank_count = 0

        while True:
            # 2. 인접 쌍 빈도 계산 (vocab이 딕셔너리이므로 .items() 호출 가능)
            pairs = self.get_adjacent_pairs(vocab)

            # 더 이상 병합할 쌍이 없으면 종료 (empty Counter는 False 취급)
            if not pairs:
                break

            # 3. 가장 빈도가 높은 쌍 찾기
            best_pair = max(pairs, key=pairs.get)
            freq = pairs[best_pair]

            if freq < self.min_frequency:
                print(
                    f"Stop training: Best pair {best_pair} freq {freq} < min_freq {self.min_frequency}"
                )
                break

            # 4. 병합 수행 (딕셔너리 -> 딕셔너리)
            vocab, new_token = self.merge_pair(best_pair, vocab)

            # 5. update the merge dictionary and vocab
            self.merges[best_pair] = new_token
            self.ranks[best_pair] = rank_count
            rank_count += 1
            if new_token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[new_token] = idx
                self.idx2word[idx] = new_token

            print(f"Merged: {best_pair} -> {new_token} (Freq: {freq})")

        self.vocab_size = len(self.word2idx)
        print(f"Training completed. Vocab size: {self.vocab_size}")
        return self.word2idx, vocab

    def encode(self, text: str):
        encoded_ids = []
        encoded_tokens = []

        for word in text.strip().split():
            word_tokens = list(word) + ["</w>"]

            while len(word_tokens) > 1:
                # 현재 단어 내의 모든 인접 쌍 추출
                pairs = [
                    (word_tokens[i], word_tokens[i + 1])
                    for i in range(len(word_tokens) - 1)
                ]

                # 2. merge rule에 있는 쌍만 필터링
                candidate_pairs = [p for p in pairs if p in self.ranks]

                if not candidate_pairs:
                    break

                best_pair = min(candidate_pairs, key=lambda x: self.ranks[x])

                # 3. merge the best pair and replace the word tokens
                new_word_tokens = []
                i = 0

                while i < len(word_tokens):
                    if (
                        i < len(word_tokens) - 1
                        and (word_tokens[i], word_tokens[i + 1]) == best_pair
                    ):
                        # self.merges에서 해당 토큰 가져오기
                        new_word_tokens.append(self.merges[best_pair])
                        i += 2
                    else:
                        new_word_tokens.append(word_tokens[i])
                        i += 1

                word_tokens = new_word_tokens

            # 결과 저장
            encoded_tokens.extend(word_tokens)
            # id 변환 (unknown token은 <unk>로 변환)
            ids = [
                self.word2idx.get(token, self.word2idx["<unk>"])
                for token in word_tokens
            ]
            encoded_ids.extend(ids)

        return encoded_ids, encoded_tokens


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
    corpus = ["low lower newest widest"]
    tokenizer = BPETokenizer()
    word2idx, words = tokenizer.train(corpus)
    print(f"Word2Idx: {word2idx}")
    print(f"Words: {words}")
    encoded_ids, encoded_tokens = tokenizer.encode("low lower newest widest highest")
    print(f"Encoded IDs: {encoded_ids}")
    print(f"Encoded Tokens: {encoded_tokens}")
