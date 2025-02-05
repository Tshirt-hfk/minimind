import os
import random
import torch
import warnings
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import Transformer
from model.LMConfig import LMConfig
from accelerate import load_checkpoint_and_dispatch

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config, model_path=None, model_from=1):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    if model_from == 1:
        model = Transformer(lm_config)
        model = load_checkpoint_and_dispatch(model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained('minimind', trust_remote_code=True)
    model = model.to(device)

    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')
    return model, tokenizer


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    seed = random.randint(1, 2000)
    # device = 'cuda:0'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    lm_config = LMConfig()
    out_dir = './out/full_sft/'
    # -----------------------------------------------------------------------------

    model, tokenizer = init_model(lm_config, out_dir)
    model = model.eval()

    # 消息模板，具体实现根据你的tokenizer进行调整
    messages_origin = [{"role": "system", "content": "开始回答问题"}]

    # 定义文件目录
    File_Dir = "ceval/ceval-exam/val"
    results_dir = "ceval/ceval_result"

    # 确保结果目录存在
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 用于记录所有文件的总正确数和总题数
    total_correct = 0
    total_questions = 0

    # 遍历目录下的所有CSV文件
    for filename in os.listdir(File_Dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(File_Dir, filename)
            test_df = pd.read_csv(file_path)

            # 存储结果的DataFrame
            result = []
            total_correct_in_file = 0  # 用于记录当前文件的正确数

            for row in test_df.itertuples(index=True, name='Pandas'):
                id = getattr(row, 'id')
                question = getattr(row, 'question')
                A = getattr(row, 'A')
                B = getattr(row, 'B')
                C = getattr(row, 'C')
                D = getattr(row, 'D')
                right_answer = getattr(row, 'answer')

                prompt = f'{question}。选择 A: {A}, B: {B}, C: {C}, D: {D}'

                messages = messages_origin.copy()
                messages.append({"role": "user", "content": prompt})

                # print(messages)
                new_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                x = tokenizer(new_prompt).data['input_ids']
                x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
                res_ids = model.eval_answer(x)

                # 假设 res_ids 是模型的 logits 输出，我们使用 softmax 转换为概率分布
                probabilities = F.softmax(res_ids, dim=-1)

                # 定义每个选项的 token id
                A_id = tokenizer('A').data['input_ids']
                B_id = tokenizer('B').data['input_ids']
                C_id = tokenizer('C').data['input_ids']
                D_id = tokenizer('D').data['input_ids']

                # 获取每个选项的概率
                A_prob = probabilities[0, A_id].item()
                B_prob = probabilities[0, B_id].item()
                C_prob = probabilities[0, C_id].item()
                D_prob = probabilities[0, D_id].item()

                # 将每个选项的概率放入字典中便于处理
                options_prob = {
                    'A': A_prob,
                    'B': B_prob,
                    'C': C_prob,
                    'D': D_prob
                }

                # 找到具有最大概率的选项
                max_option_answer = max(options_prob, key=options_prob.get)

                # 比较答案并记录
                is_right = 1 if max_option_answer == right_answer else 0
                result.append({
                    'question': question,
                    'A': A,
                    'B': B,
                    'C': C,
                    'D': D,
                    'answer': right_answer,
                    'llm_answer': max_option_answer,
                    'is_right': is_right
                })
                # print(f'id: {id} 问题: {question[:10]}... 是否正确: {is_right}')

                if is_right:
                    total_correct_in_file += 1

            total_correct += total_correct_in_file
            total_questions += len(test_df)

            # 计算当前文件的正确率并添加到结果DataFrame的最后一行
            accuracy = total_correct_in_file / len(test_df)
            result.append({
                'question': '-',
                'A': '-',
                'B': '-',
                'C': '-',
                'D': '-',
                'answer': f'文件 {filename} 的正确率: {accuracy:.2%}',
                'llm_answer': '-',
                'is_right': '-'
            })

            print(f'{filename.split(".")[0]} ，{total_correct_in_file}/{len(test_df)}，正确率: {accuracy:.2%}')

            # 保存结果到CSV
            results_path = os.path.join(results_dir, f"{filename.split('.')[0]}_result.csv")
            results_df = pd.DataFrame(result)
            results_df.to_csv(results_path, index=False)

    # 计算总正确率
    total_accuracy = total_correct / total_questions if total_questions > 0 else 0

    # 将各个文件的正确率以及总正确率写入到 "ceval/ceval_result/test.log"
    log_path = os.path.join(results_dir, "test.log")
    with open(log_path, 'w') as log_file:
        result = f"总题数: {total_questions}\n总正确数: {total_correct}\n总正确率: {total_accuracy:.2%}"
        log_file.write(result)

        print(result)

        for filename in os.listdir(File_Dir):
            if filename.endswith('.csv'):
                accuracy_file = pd.read_csv(os.path.join(results_dir, f"{filename.split('.')[0]}_result.csv"))
                last_row = accuracy_file.iloc[-1]['answer']
                log_file.write(f"{filename}: {last_row}\n")
