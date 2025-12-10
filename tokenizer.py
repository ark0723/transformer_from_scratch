import re
import collections
import numpy as np


class BPETokenizer:
    def __init__(
        self,
        vocab_size: int = 10000,
        special_tokens: list[str] = ["<pad>", "<unk>", "<sos>", "<eos>"],
    ):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # initialize vocabulary
        self.word2idx = {word: i for i, word in enumerate(self.special_tokens)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}

        # pre-save special token IDs for efficiency
        self.pad_token_id = self.word2idx["<pad>"]
        self.unk_token_id = self.word2idx["<unk>"]
        self.sos_token_id = self.word2idx["<sos>"]
        self.eos_token_id = self.word2idx["<eos>"]

        # BPE merge dictionary
        self.merges = {}  # merge rules for encoding : pair -> new_token
        self.ranks = {}  # pair -> rank

    def initialize(self, corpus: list[str]) -> tuple[dict, dict]:
        """
        Converts corpus into word counts and initial vocab list.
        Example: {'low': 5} -> vocab_list {'low': ['l', 'o', 'w', '</w>']}
        """
        # 1. 모든 단어를 분리하여 카운트
        word_counts = collections.Counter()
        for sentence in corpus:
            # 2. 각 문장을 space 기준으로 단어로 분리하고 각 단어를 카운트
            words = sentence.strip().split()
            word_counts.update(words)

        # 단어를 문자 리스트로 변환 (끝 문자 표시)
        # vocab: { "word": ["c", "h", "a", "r", "s", "</w>"] }
        vocab_list = {word: list(word) + ["</w>"] for word in word_counts}
        return word_counts, vocab_list

    def get_stats(self, vocab_list, word_counts):
        """
        Calculates pair frequencies and creates an inverted index.
        """
        pairs = collections.defaultdict(int)
        # inverted_index: pair -> {word_original_string, ...}
        # 해당 쌍이 포함된 '단어(key)'들의 집합(Set)을 저장
        inverted_index = collections.defaultdict(set)

        for word, symbols in vocab_list.items():
            freq = word_counts[word]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
                inverted_index[pair].add(word)

        return pairs, inverted_index

    def train(self, corpus: list[str]):
        # 1. 초기화
        word_counts, vocab_list = self.initialize(corpus)

        # 2. 초기 쌍 통계 및 역색인 생성 (최초 1회만 수행)
        pairs, inverted_index = self.get_stats(vocab_list, word_counts)

        rank_counter = 0

        # 3. 목표 vocab size 도달 시까지 반복
        # (현재 단어장 크기 = 초기 특수토큰 수 + 병합된 횟수)
        current_vocab_size = len(self.word2idx)

        # add base characters to vocab first
        # This ensures all single chars are known even if not merged
        initial_chars = set()
        for symbols in vocab_list.values():
            initial_chars.update(symbols)

        for char in sorted(list(initial_chars)):
            if char not in self.word2idx:
                new_id = len(self.word2idx)
                self.word2idx[char] = new_id
                self.idx2word[new_id] = char
                current_vocab_size += 1

        while current_vocab_size < self.vocab_size:
            if not pairs:
                break

            # 3. 가장 빈도가 높은 쌍 찾기
            best_pair = max(pairs, key=pairs.get)
            freq = pairs[best_pair]

            # 4. 병합 정보 저장
            new_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = new_token
            self.ranks[best_pair] = rank_counter

            if new_token not in self.word2idx:
                new_id = len(self.word2idx)
                self.word2idx[new_token] = new_id
                self.idx2word[new_id] = new_token
                current_vocab_size += 1

            rank_counter += 1
            # print(f"Merged: {best_pair} -> {new_token} (Freq: {freq})")

            # === [Core Logic: Subtract -> Merge -> Add] ===

            # 1. best_pair를 포함하는 단어들의 목록을 복사 (Set 순회 중 수정 방지)
            words_to_update = list(inverted_index[best_pair])

            # 2. 해당 단어들에 대해 반복
            for word in words_to_update:
                current_freq = word_counts[word]
                current_symbols = vocab_list[word]

                # [Step A: Subtract] 현재 상태의 모든 쌍 정보를 제거
                # (단어 하나는 길어야 수십 글자이므로 전체를 지웠다 다시 쓰는 게 훨씬 안전하고 빠름)
                for i in range(len(current_symbols) - 1):
                    p = (current_symbols[i], current_symbols[i + 1])
                    pairs[p] -= current_freq
                    if pairs[p] == 0:
                        del pairs[p]
                    # 역색인에서도 제거 (안전하게 discard 사용)
                    if p in inverted_index:
                        inverted_index[p].discard(word)

                # [Step B: Merge] 단어 리스트 내부 병합 수행 (replace)
                new_symbols = []
                i = 0
                while i < len(current_symbols):
                    if (
                        i < len(current_symbols) - 1
                        and (current_symbols[i], current_symbols[i + 1]) == best_pair
                    ):
                        new_symbols.append(new_token)
                        i += 2
                    else:
                        new_symbols.append(current_symbols[i])
                        i += 1

                # vocab_list 업데이트
                vocab_list[word] = new_symbols

                # [Step C: Add] 새로운 상태의 모든 쌍 정보를 다시 등록
                for i in range(len(new_symbols) - 1):
                    p = (new_symbols[i], new_symbols[i + 1])
                    pairs[p] += current_freq
                    inverted_index[p].add(word)

            # (참고) best_pair는 위 로직에 의해 자연스럽게 pairs와 inverted_index에서 빈도가 0이 되거나 제거됨
            # 혹시 남아있는 쓰레기 값이 있다면 정리
            if best_pair in pairs:
                del pairs[best_pair]
            if best_pair in inverted_index:
                del inverted_index[best_pair]

        print(f"Training completed. Final vocab size: {len(self.word2idx)}")
        return self.word2idx

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False):
        """
        Encodes a single string into a list of integers.
        Added add_sos/add_eos args for compatibility with Seq2SeqCollator.
        """
        encoded_ids = []

        # Split text into words to handle BPE per word
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

            # Convert final tokens to IDs
            # If a subword/character is unknown (not in vocab), map to <unk>
            ids = [self.word2idx.get(token, self.unk_token_id) for token in word_tokens]
            encoded_ids.extend(ids)

        if add_sos:
            encoded_ids = [self.sos_token_id] + encoded_ids
        if add_eos:
            encoded_ids = encoded_ids + [self.eos_token_id]

        return encoded_ids

    def decode(
        self,
        indices: list[int] | list[list[int]] | np.ndarray,
        skip_special_tokens: bool = False,
    ) -> list[str] | str:
        """
        Decodes a list of IDs back to a string.
        Handles BPE specific logic (removing </w>).
        """

        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if isinstance(indices[0], list):
            return [self.decode(seq, skip_special_tokens) for seq in indices]

        tokens = []
        for idx in indices:
            token = self.idx2word.get(idx, "<unk>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)

        # join tokens to from the rough string
        text = "".join(tokens)

        # Post-processing for BPE: replace </w> with space
        # </w> indicates end of word, so we replace it with space
        text = text.replace("</w>", " ").strip()

        return text

    def __call__(
        self,
        text: str | list[str],
        max_length: int | None = None,
        add_sos: bool = False,
        add_eos: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Batch encoding interface for DataLoader
        """
        if isinstance(text, str):
            text = [text]

        # 1. Encode all sentences
        batch_ids = [
            self.encode(sentence, add_sos=add_sos, add_eos=add_eos) for sentence in text
        ]

        # 2. Padding
        batch_size = len(batch_ids)

        # if max_length is not provided, use the longest sequence in the batch
        if max_length is None:
            max_length = max(len(ids) for ids in batch_ids)

        input_ids = np.full((batch_size, max_length), self.pad_token_id, dtype=np.int32)
        attention_mask = np.zeros((batch_size, max_length), dtype=np.int32)

        for i, ids in enumerate(batch_ids):
            length = min(len(ids), max_length)
            input_ids[i, :length] = ids[:length]
            attention_mask[i, :length] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}


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
    corpus = ["low lower newest widest highest fastest"]
    tokenizer = BPETokenizer(vocab_size=50)
    word2idx = tokenizer.train(corpus)

    print("\n--- Encoding Test ---")
    test_str = "lowest highest"
    encoded_ids = tokenizer.encode(test_str, add_sos=True, add_eos=True)
    print(f"Input: '{test_str}'")
    print(f"Encoded IDs: {encoded_ids}")

    print("\n--- Decoding Test ---")
    decoded_str = tokenizer.decode(encoded_ids, skip_special_tokens=True)
    print(f"Decoded: '{decoded_str}'")

    print("\n--- Batch Test (__call__) ---")
    batch_input = ["low lower", "widest"]
    output = tokenizer(batch_input, max_length=10, add_sos=True)
    print("Batch Input:", batch_input)
    print("Input IDs:\n", output["input_ids"])
    print("Attention Mask:\n", output["attention_mask"])
