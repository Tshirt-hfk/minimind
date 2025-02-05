
import os
import json
import ujson
import random
import itertools
import jsonlines
import pyarrow as pa
import pyarrow.parquet as pq
from unicodedata import normalize
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

random.seed(42)


def process_tokenizer_data(target_data_path, chunk_size=50000):

    def handle_seq_monkey_data(data_path, sample_rate=1.0):
        name = data_path.split("/")[-1].split(".")[0]
        lines = []
        with jsonlines.open(data_path) as reader:       
            chunk_idx = 0
            while True:
                chunk = list(itertools.islice(reader, chunk_size))
                if not chunk:
                    break
                for idx, obj in enumerate(chunk):
                    try:
                        if random.random() > sample_rate:
                            continue
                        content = obj.get('text', '')
                        if not content:
                            continue
                        lines.append(content)
                    except UnicodeDecodeError as e:
                        print(f"Skipping invalid line {chunk_idx * chunk_size + idx + 1}: {e}")
                        continue
        print(f"{name} done.")
        return lines
 
    def handle_wikipedia_data(data_path, sample_rate=1.0):
        name = data_path.split("/")[-1].split(".")[0]
        lines = []
        with open(data_path, "r", encoding="utf-8") as f:
            items = ujson.load(f)
            for item in items:
                if random.random() > sample_rate:
                    continue
                lines.append(item["completion"])
        print(f"{name} done.")
        return lines
    
    def handle_baidubaike_data(data_path, sample_rate=1.0):
        name = data_path.split("/")[-1].split(".")[0]
        lines = []
        with open(data_path, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if random.random() > sample_rate:
                    continue
                item = ujson.loads(line)
                if not item["title"]:
                    continue
                line = item['title']
                if item['summary']:
                    line = f"{line}：{item['summary']}"
                # for section in item["sections"]:
                #     if section["title"] and section["content"]:
                #         line = f"{line}\n{section['title']}：{section['content']}"
                #     elif section["title"]:
                #         line = f"{line}\n{section['title']}"
                #     elif section["content"]:
                #         line = f"{line}\n{section['content']}"
                line = normalize("NFKC", line)
                lines.append(line)
        print(f"{name} done.")
        return lines

    def handle_sky_data(data_folder, sample_rate=1.0):
        lines = []
        for filename in os.listdir(data_folder):
            if filename.endswith(".jsonl"):
                name = filename.split(".")[0]
                with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
                    for line in f:
                        if random.random() > sample_rate:
                            continue
                        item = ujson.loads(line)
                        lines.append(item["text"])
                print(f"{name} done.")
        return lines
    
    lines = []
    # lines.append(handle_seq_monkey_data('./dataset/raw_data/pretrain_data/mobvoi_seq_monkey_general_open_corpus.jsonl'))
    lines.extend(handle_wikipedia_data('./dataset/raw_data/pretrain_data/wikipedia-cn-20230720-filtered.json'))
    lines.extend(handle_baidubaike_data('./dataset/raw_data/pretrain_data/563w_baidubaike.json'))
    lines.extend(handle_sky_data('./dataset/raw_data/pretrain_data/SkyPile-150B', sample_rate=0.1))
    tb = pa.Table.from_arrays([pa.array(lines)], names=["text"])
    pq.write_table(
        table=tb,
        where=target_data_path,
        row_group_size=50000,
        data_page_size=50000,
    )


def train_tokenizer(data_path, tokenizer_dir):

    def read_texts_from_parquet(data_path):
        df = pq.read_table(data_path).to_pandas()
        texts = df['text'].tolist()
        return texts

    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True), # 数字单独分词
        pre_tokenizers.ByteLevel(add_prefix_space=False) # 字符级别分词
    ])

    # 定义特殊token
    special_tokens = ["<pad>", "<s>", "</s>"]

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=64000,
        special_tokens=special_tokens,  # 确保这三个token被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() # 使用ByteLevel的默认字符集，确保无UNK
    )

    # 读取文本数据
    texts = read_texts_from_parquet(data_path)

    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<pad>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    # 保存tokenizer
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            "0": {
                "content": "<pad>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<pad>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": None,
        "use_default_system_prompt": False,
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content']+'\\n' %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user:' + content + '</s>\\n<s>assistant:' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")


def eval_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '⚪是圆形的'},
        {"role": "assistant", "content": '456'},
        {"role": "user", "content": '456'},
        {"role": "assistant", "content": '789'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    print(new_prompt)
    # 获取词汇表大小（不包括特殊符号）
    print('tokenizer词表大小：', tokenizer.vocab_size)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('qwen实际词表长度：', actual_vocab_size)

    new_prompt = 'wenjie，椭圆和⚪的关系是什么呢？因为明天下午要带家人去下医院，所以申请上午在家办公，因为明天下午要带家人去下医院，所以申请上午在家办公，因为明天下午要带家人去下医院，所以申请上午在家办公，下午请半天假~@LWJWe '
    print(new_prompt)
    model_inputs = tokenizer(new_prompt)

    print(model_inputs)
    print('长度：', len(model_inputs['input_ids']))

    input_ids_ = model_inputs['input_ids']

    response = tokenizer.decode(input_ids_)
    print(response, end='')


def main():
    # process_tokenizer_data('./dataset/tokenizer_data/train_tokenizer.parquet')
    train_tokenizer('./dataset/tokenizer_data/train_tokenizer.parquet', "./model/minimind_tokenizer")
    eval_tokenizer("./model/minimind_tokenizer")


if __name__ == '__main__':
    main()
