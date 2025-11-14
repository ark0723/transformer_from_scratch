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
