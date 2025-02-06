import os
import time
import torch
import warnings
import argparse
import torch.nn.functional as F
from copy import deepcopy
from torch import optim
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin, load_checkpoint_and_dispatch
from accelerate.utils import gather_object
from transformers import AutoTokenizer, set_seed, get_constant_schedule_with_warmup
from model.model import Transformer
from model.LMConfig import LMConfig
from model.dataset import DPODataset

warnings.filterwarnings('ignore')


def dpo_loss_func(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=0.1):
    policy_logps = policy_chosen_logps - policy_rejected_logps
    reference_logps = reference_chosen_logps - reference_rejected_logps
    logits = policy_logps - reference_logps
    loss = -F.logsigmoid(beta * logits)
    chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()
    return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def init_model(args, lm_config, accelerator):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = Transformer(lm_config).to(accelerator.device)
    accelerator.print(model)
    if args.load_dir is not None:
        model = load_checkpoint_and_dispatch(model, args.load_dir)
        accelerator.print(f'loading model from {args.load_dir}')
    accelerator.print(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


def init_data(args, lm_config, tokenizer, accelerator):
    data_paths = []
    for name in os.listdir(args.data_dir):
        if name.endswith('.parquet'):
            data_path = os.path.join(args.data_dir, name)
            data_paths.append(data_path)
    accelerator.print(f"loading datasets: {data_paths}")
    train_ds = DPODataset(data_paths, tokenizer, max_length=lm_config.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers
    )
    return train_loader


def train(args, lm_config):
    deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=args.accumulation_steps, gradient_clipping=args.grad_clip, zero_stage=2)
    accelerator = Accelerator(mixed_precision='bf16', deepspeed_plugin=deepspeed_plugin)
    model, tokenizer = init_model(args, lm_config, accelerator)
    train_loader = init_data(args, lm_config, tokenizer, accelerator)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps
    )
    ref_model = deepcopy(model)
    ref_model.eval()
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    if args.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None
    start_time = time.time()
    update_step = 0
    total_update_steps = len(train_loader) * args.epochs // args.accumulation_steps
    accumulation_loss = 0
    chosen_reward = 0
    rejected_reward = 0
    model.train()
    for epoch in range(args.epochs):
        for step, ((chosen_X, chosen_Y, chosen_loss_mask), (rejected_X, rejected_Y, rejected_loss_mask)) in enumerate(train_loader):

            chosen_X, rejected_X = chosen_X.to(accelerator.device), rejected_X.to(accelerator.device)
            chosen_Y, rejected_Y = chosen_Y.to(accelerator.device), rejected_Y.to(accelerator.device)
            chosen_loss_mask, rejected_loss_mask = chosen_loss_mask.to(accelerator.device), rejected_loss_mask.to(accelerator.device)

            with torch.no_grad():
                reference_chosen_out = ref_model(chosen_X, chosen_Y)
                reference_rejected_out = ref_model(rejected_X, rejected_Y)
                reference_chosen_logps = -(reference_chosen_out.last_loss.view(chosen_loss_mask.size()) * chosen_loss_mask).sum(dim=-1) / chosen_loss_mask.sum(dim=-1)
                reference_rejected_logps = -(reference_rejected_out.last_loss.view(rejected_loss_mask.size()) * rejected_loss_mask).sum(dim=-1) / rejected_loss_mask.sum(dim=-1)

            policy_chosen_out = model(chosen_X, chosen_Y)
            policy_rejected_out = model(rejected_X, rejected_Y)
            policy_chosen_logps = -(policy_chosen_out.last_loss.view(chosen_loss_mask.size()) * chosen_loss_mask).sum(dim=-1) / chosen_loss_mask.sum(dim=-1)
            policy_rejected_logps = -(policy_rejected_out.last_loss.view(rejected_loss_mask.size()) * rejected_loss_mask).sum(dim=-1) / rejected_loss_mask.sum(dim=-1)
            
            loss, chosen_reward, rejected_reward = dpo_loss_func(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
            loss = loss / args.accumulation_steps
            chosen_reward += chosen_reward.item() / args.accumulation_steps
            rejected_reward += rejected_reward.item() / args.accumulation_steps
            accumulation_loss += loss.item()
            accelerator.backward(loss)
            if (step + 1) % args.accumulation_steps == 0:
                if update_step % args.log_interval == 0:
                    accumulation_loss = sum(gather_object([accumulation_loss])) / accelerator.num_processes
                    spend_time = time.time() - start_time
                    accelerator.print('Epoch:[{}/{}]({}/{}) loss:{:.3f} chosen_reward:{:.3f} rejected_reward:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                            epoch, args.epochs, update_step, total_update_steps, accumulation_loss,
                            chosen_reward, rejected_reward, optimizer.param_groups[-1]['lr'],
                            (spend_time / (update_step + 1) * total_update_steps - spend_time) // 60
                        )
                    )
                    if wandb is not None and accelerator.is_main_process:
                        wandb.log({"loss": accumulation_loss, "lr": optimizer.param_groups[-1]['lr'],
                                "chosen_reward": chosen_reward, "rejected_reward": rejected_reward,
                                "epoch_Time": (spend_time / (update_step + 1) * total_update_steps - spend_time) // 60})
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                update_step += 1
                accumulation_loss = 0
                chosen_reward = 0
                rejected_reward = 0
                if (update_step % args.save_interval == 0 or update_step == total_update_steps) and accelerator.is_main_process:
                    accelerator.save_model(model, args.save_dir)


# PORTS=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
# accelerate launch --multi_gpu --num_processes 8 --main_process_port ${PORTS[0]} 2-dpo_train.py --use_wandb
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="Weights & Biases project name")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--data_dir", type=str, default="./dataset/dpo_data/", help="Path to training data")
    parser.add_argument("--load_dir", type=str, default="./out/sharedkv/full_sft/", help="Path to loading model")
    parser.add_argument("--save_dir", type=str, default="./out/sharedkv/dpo_train/", help="Path to saving model")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval")

    args = parser.parse_args()
    lm_config = LMConfig()
    args.wandb_run_name = f"Transformer-SharedKV-Length_{lm_config.max_seq_len}-Layer_{lm_config.n_layers}-Dim_{lm_config.dim}"
    torch.manual_seed(1337)
    set_seed(1337)

    train(args=args, lm_config=lm_config)
