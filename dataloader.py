from torch.utils.data import Dataset
import torch
from tokenizer import Tokenizer


class TextDataset(Dataset):
    """
    A custom Dataset class to handle the raw text corpus.
    It simply stores the list of strings and retrieves them by index.

    Args:
        corpus: list of strings
    """

    def __init__(self, corpus: list[str]):
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        """
        Retrieves the raw text string at the specified index.
        """
        return self.corpus[idx]


class TranslationDataset(Dataset):
    """
    Dataset for Seq2Seq (Source -> Target).
    Returns raw strings for source and target sentences.
    """

    def __init__(self, src_corpus: list[str], tgt_corpus: list[str]):
        assert len(src_corpus) == len(
            tgt_corpus
        ), "Source and target corpus must have the same length"
        self.src_corpus = src_corpus
        self.tgt_corpus = tgt_corpus

    def __len__(self):
        return len(self.src_corpus)

    def __getitem__(self, idx):
        """
        Retrieves the raw text strings for source and target at the specified index.
        """
        return self.src_corpus[idx], self.tgt_corpus[idx]


class DataCollator:
    """
    A collator function object to be used with DataLoader's `collate_fn`.
    It takes a list of raw samples and converts them into a single batched tensor.
    """

    def __init__(self, tokenizer: Tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[str]):
        """
        Processes a batch of text samples.

        Args:
            batch: A list of raw strings retrieved from the Dataset.

        Returns:
            A dictionary containing:
            - 'input_ids': (batch_size, max_length) LongTensor
            - 'attention_mask': (batch_size, max_length) LongTensor
        """
        # 1. Perform batch encoding and padding using the tokenizer's __call__ method
        encoded_batch = self.tokenizer(batch, max_length=self.max_length)

        # 2. Convert the NumPy array to a PyTorch Tensor
        input_ids = torch.tensor(encoded_batch["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded_batch["attention_mask"], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class Seq2SeqCollator:
    """
    A collator for Seq2Seq (Source -> Target).
    Handles tokenization, padding, and special token injection (<sos>, <eos>).
    """

    def __init__(
        self, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer, max_length: int = 20
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[tuple[str, str]]):
        """
        Processes a batch of source and target text pairs.

        Args:
            batch: A list of tuples containing source and target text pairs.

        Returns:
            A dictionary containing:
            - 'input_ids': (batch_size, max_length) LongTensor
            - 'attention_mask': (batch_size, max_length) LongTensor
            - 'decoder_input_ids': (batch_size, max_length) LongTensor
            - 'decoder_attention_mask': (batch_size, max_length) LongTensor
        """
        src_sentences, tgt_sentences = zip(*batch)

        # 1. Source Encoding (Encoder Input)
        # Tokenizer.__call__ returns {'input_ids': np.array, 'attention_mask': np.array}
        src_encoded = self.src_tokenizer(
            src_sentences, max_length=self.max_length, add_sos=False, add_eos=False
        )

        # 2. Target Encoding (Decoder Input)
        tgt_input_encoded = self.tgt_tokenizer(
            tgt_sentences, max_length=self.max_length, add_sos=True, add_eos=False
        )
        # 3. Target Label Encoding (Decoder Output)
        tgt_label_encoded = self.tgt_tokenizer(
            tgt_sentences, max_length=self.max_length, add_sos=False, add_eos=True
        )

        return {
            "src_ids": torch.tensor(src_encoded["input_ids"], dtype=torch.long),
            "src_mask": torch.tensor(src_encoded["attention_mask"], dtype=torch.long),
            "tgt_input_ids": torch.tensor(
                tgt_input_encoded["input_ids"], dtype=torch.long
            ),
            "tgt_mask": torch.tensor(
                tgt_input_encoded["attention_mask"], dtype=torch.long
            ),
            "tgt_label_ids": torch.tensor(
                tgt_label_encoded["input_ids"], dtype=torch.long
            ),
        }
