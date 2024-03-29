import torch
import torch.nn as nn
from torch.utils.data import Dataset # data loader

from kogpt2_transformers import get_kogpt2_tokenizer
from kobert_transformers import get_tokenizer

class AutoRegressiveDataset(Dataset):
  """Auto Regressive Dataset"""

  def __init__(self,
               file_path = "../data/Wellness.re.tsv",
               n_ctx = 1024
               ):
    self.file_path = file_path
    self.data =[]
    self.tokenizer = get_kogpt2_tokenizer()


    bos_token_id = [self.tokenizer.bos_token_id]
    eos_token_id = [self.tokenizer.eos_token_id]
    pad_token_id = [self.tokenizer.pad_token_id]

    file = open(self.file_path, 'r', encoding='utf-8')

    while True:
      line = file.readline()
      if not line:
        break
      datas = line.split('\t')
      index_of_words = bos_token_id +self.tokenizer.encode(datas[0]) + eos_token_id + bos_token_id + self.tokenizer.encode(datas[1][:-1])+ eos_token_id
      pad_token_len = n_ctx - len(index_of_words)

      index_of_words += pad_token_id * pad_token_len

      self.data.append(index_of_words)

    file.close()

  def __len__(self):
    return len(self.data)

  def __getitem__(self,index):
    item = self.data[index]
    return item

if __name__ == "__main__":
  dataset = AutoRegressiveDataset()
  print(dataset)
