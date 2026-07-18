import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token
        num = 4
        list_word = []
        for i in texts:
            list_str = i.lower().split()
            list_word.extend(list_str)
        uniqueWord = list(Dict.fromkeys(list_word))
        uniqueWord.sort()
        for i in uniqueWord:
            self.word_to_id[i] = num
            self.id_to_word[num] =  i
            num += 1
        self.vocab_size = num
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        list_str = text.lower().split()
        list = []
        for i in list_str:
            if i not in self.word_to_id:
                list.append(1)
                continue   
            list.append(self.word_to_id[i])
        return list
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        str = ""
        for i in ids:
            if i not in self.id_to_word:
                str += self.unk_token
                str += " "
                continue
            str += self.id_to_word[i]
            str += " "
        return str.strip()
        
