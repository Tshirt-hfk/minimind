import os
import torch
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def find_sublist_index(main_list, sub_list) -> int:
    last_index = -1
    for i in range(len(main_list) - len(sub_list) + 1):
        if main_list[i:i + len(sub_list)] == sub_list:
            last_index = i
    return last_index


class PretrainDataset(Dataset):
    def __init__(self, data_paths, tokenizer, max_length=1024):
        super().__init__()
        self.df = pd.concat([pq.read_table(data_path).to_pandas() for data_path in data_paths], ignore_index=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        text = str(sample['text'])
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


class SFTDataset(Dataset):
    def __init__(self, data_paths, tokenizer, max_length=1024, prompt_max_len=512, answer_max_len=256):
        super().__init__()
        self.df = pd.concat([pq.read_table(data_path).to_pandas() for data_path in data_paths], ignore_index=True)
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        self.tokenizer = tokenizer
        self.padding = tokenizer.pad_token_id
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        instruction = str(sample['instruction'])
        history = sample['history']
        question = str(sample['question'])
        answer = str(sample['answer'])

        messages = []
        if instruction:
            messages.append(
                {"role": 'system', "content": instruction}
            )
        for history_message in history:
            if len(history_message) <= 1:
                continue
            messages.append(
                {"role": 'user', "content": str(history_message[0])}
            )
            messages.append(
                {"role": 'assistant', "content": str(history_message[1])}
            )
        messages += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_id = self.tokenizer(new_prompt).data['input_ids'][:self.max_length]

        # 实际长度
        question_length = find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - len(input_id)
        input_id = input_id + [self.padding] * padding_len
        mask_len = len(input_id) - question_length - padding_len
        # 0表示不计算损失
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)

        return X_tensor, Y_tensor, loss_mask_tensor


class DPODataset(Dataset):
    def __init__(self, data_paths, tokenizer, max_length=1024):
        super().__init__()
        self.df = pd.concat([pq.read_table(data_path).to_pandas() for data_path in data_paths], ignore_index=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        prompt = str(sample['prompt'])
        rejected = str(sample['rejected'])
        chosen = str(sample['chosen'])
        results = []
        for answer in [chosen, rejected]:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            input_id = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            input_id = self.tokenizer(input_id).data['input_ids'][:self.max_length]
            # 实际长度
            question_length = find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
            # 没满最大长度的剩余部分
            padding_len = self.max_length - len(input_id)
            input_id = input_id + [self.padding] * padding_len
            mask_len = len(input_id) - question_length - padding_len
            # 0表示不计算损失
            loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len

            input_id = np.array(input_id)
            X = np.array(input_id[:-1]).astype(np.int64)
            Y = np.array(input_id[1:]).astype(np.int64)
            loss_mask = np.array(loss_mask[1:]).astype(np.int64)

            X_tensor = torch.from_numpy(X)
            Y_tensor = torch.from_numpy(Y)
            loss_mask_tensor = torch.from_numpy(loss_mask)
            results.append((X_tensor, Y_tensor, loss_mask_tensor))
        return results

if __name__ == "__main__":
    pass
