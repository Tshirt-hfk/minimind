import os
import re
import ujson
import jsonlines
import itertools
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from unicodedata import normalize
from transformers import AutoTokenizer

bos_token = "<s>"
eos_token = "</s>"
# tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer', use_fast=False)
# print('tokenizer词表大小：', len(tokenizer))


def split_txt_cropus_to_chunk_data(
    texts: list, max_len: int = 1024, window_size: int = 2, batch_size: int = 512**2
) -> list:

    buffer, buffer_len = [], 0
    chunk_data = []

    for i, line in enumerate(texts):
        buffer_len += len(line)
        buffer.append(line)

        if buffer_len >= batch_size or i == len(texts) - 1:
            buffer_txt = "".join(buffer)

            # - window_size为滑动窗口，这样每个窗口都包含有window_size个上文
            for i in range(0, len(buffer_txt), max_len - window_size):

                chunk_data.append("".join(buffer_txt[i : i + max_len]))

            buffer, buffer_len = [], 0

    return chunk_data


def pretrain_process(chunk_size=50000, max_len=1024, window_size=2):

    def handle_seq_monkey_data(data_path):
        name = data_path.split("/")[-1].split(".")[0]
        with jsonlines.open(data_path) as reader:       
            chunk_idx = 0
            lines = []
            while True:
                chunk = list(itertools.islice(reader, chunk_size))
                if not chunk:
                    break
                for idx, obj in enumerate(chunk):
                    try:
                        content = obj.get('text', '')
                        if not content:
                            continue
                        lines.append(content + eos_token)
                    except UnicodeDecodeError as e:
                        print(f"Skipping invalid line {chunk_idx * chunk_size + idx + 1}: {e}")
                        continue
                chunk_idx += 1

            chunk_data = split_txt_cropus_to_chunk_data(lines, max_len, window_size)
            tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
            pq.write_table(
                table=tb,
                where=f'./dataset/pretrain_data/{name}_{max_len}.parquet',
                row_group_size=50000,
                data_page_size=50000,
            )
            print(f"{name} done.")
 
    def handle_wikipedia_data(data_path):
        name = data_path.split("/")[-1].split(".")[0]
        with open(data_path, "r", encoding="utf-8") as f:
            items = ujson.load(f)
            lines = []
            for item in items:
                lines.append(item["completion"] + eos_token)
            chunk_data = split_txt_cropus_to_chunk_data(lines)
            tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
            pq.write_table(
                table=tb,
                where=f'./dataset/pretrain_data/{name}_{max_len}.parquet',
                row_group_size=50000,
                data_page_size=50000,
            )
            print(f"{name} done.")
    
    def handle_baidubaike_data(data_path):
        name = data_path.split("/")[-1].split(".")[0]
        with open(data_path, "r", encoding="utf-8") as f:
            lines = []
            while True:
                line = f.readline()
                if not line:
                    break
                item = ujson.loads(line)
                if not item["title"]:
                    continue
                line = item['title']
                if item['summary']:
                    line = f"{line}：{item['summary']}"
                for section in item["sections"]:
                    if section["title"] and section["content"]:
                        line = f"{line}\n{section['title']}：{section['content']}"
                    elif section["title"]:
                        line = f"{line}\n{section['title']}"
                    elif section["content"]:
                        line = f"{line}\n{section['content']}"
                line = normalize("NFKC", line)
                line = f"{line}{eos_token}"
                lines.append(line)

            chunk_data = split_txt_cropus_to_chunk_data(lines)
            tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
            pq.write_table(
                table=tb,
                where=f'./dataset/pretrain_data/{name}_{max_len}.parquet',
                row_group_size=50000,
                data_page_size=50000,
            )
            print(f"{name} done.")

    def handle_sky_data(data_folder):
        for filename in os.listdir(data_folder):
            if filename.endswith(".jsonl"):
                name = filename.split(".")[0]
                lines = []
                with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
                    for line in f:
                        item = ujson.loads(line)
                        lines.append(item["text"] + eos_token)
                chunk_data = split_txt_cropus_to_chunk_data(lines)
                tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
                pq.write_table(
                    table=tb,
                    where=f'./dataset/pretrain_data/{name}_{max_len}.parquet',
                    row_group_size=50000,
                    data_page_size=50000,
                )
                print(f"{name} done.")

    handle_seq_monkey_data('./dataset/raw_data/pretrain_data/mobvoi_seq_monkey_general_open_corpus.jsonl' )
    handle_wikipedia_data('./dataset/raw_data/pretrain_data/wikipedia-cn-20230720-filtered.json')
    handle_baidubaike_data('./dataset/raw_data/pretrain_data/563w_baidubaike.json')
    handle_sky_data('./dataset/raw_data/pretrain_data/SkyPile-150B')


def sft_process(max_len=1024):
    
    def chinese_ratio(text):
        # 匹配所有中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        # 中文字符数量占比
        return len(chinese_chars) / len(text) if text else 0
        
    sft_datasets = ['./dataset/raw_data/sft_data/deepctrl-sft-data/sft_data_zh.jsonl']
    instruction_list = []
    history_list = []
    question_list = []
    answer_list = []
    for path in sft_datasets:
        name = path.split('/')[-1].split('.')[0]
        with jsonlines.open(path) as reader:
            num = 0
            for idx, obj in enumerate(reader):
                try:
                    i = obj.get('instruction', '')
                    h = obj.get('history', [])
                    q = obj.get('input', '') + obj.get('q', '')
                    a = obj.get('output', '') + obj.get('a', '')
                    s = i + "".join([a+b for a, b in h]) + q + a
                    if q and a and len(s) < max_len and chinese_ratio(s) > 0.7:
                        instruction_list.append(i)
                        history_list.append(h)
                        question_list.append(q)
                        answer_list.append(a)
                        num += 1
                    if (idx + 1) % 100000 == 0:
                        print('chunk:', (idx + 1 - 100000, idx + 1), 'process', num, 'end.')
                except jsonlines.InvalidLineError as e:
                    print(f"Skipping invalid JSON line {idx + 1}: {e}")
                    continue

            tb = pa.Table.from_arrays([pa.array(instruction_list), pa.array(history_list), pa.array(question_list), pa.array(answer_list)],
                    names=['instruction', 'history', 'question', 'answer'])
            pq.write_table(
                table=tb,
                where=f'./dataset/sft_data/{name}_{max_len}.parquet',
                row_group_size=50000,
                data_page_size=50000,
            )


def rl_process(max_len=1024):
    ################
    # Dataset
    ################

    dataset_paths = [
        './dataset/raw_data/huozi_rlhf_data.csv',
    ]
    for path in dataset_paths:
        name = path.split('/')[-1].split('.')[0]
        prompt_list, rejected_list, chosen_list = [], [], []
        df = pd.read_csv(path)
        for i, row in df.iterrows():
            prompt = row.get('prompt', '')
            chosen = row.get('chosen', '')
            reject = row.get('reject', '')
            if prompt and reject and chosen and len(prompt + chosen) < max_len and len(prompt + reject) < max_len:
                prompt_list.append(prompt)
                chosen_list.append(chosen)
                rejected_list.append(reject)

        tb = pa.Table.from_arrays([pa.array(prompt_list), pa.array(rejected_list), pa.array(chosen_list)],
                            names=['prompt', 'rejected', 'chosen'])
        pq.write_table(
            table=tb,
            where=f'./dataset/dpo_data/{name}_{max_len}.parquet',
            row_group_size=50000,
            data_page_size=50000,
        )


if __name__ == "__main__":

    ################
    # 1: pretrain
    # 2: sft
    # 3: RL
    ################
    process_type = 1

    if process_type == 1:
        pretrain_process()
    if process_type == 2:
        sft_process()
    if process_type == 3:
        rl_process()
