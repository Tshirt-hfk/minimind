import os
import time
import torch
import warnings
import argparse
from torch import optim
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin, load_checkpoint_and_dispatch
from accelerate.utils import gather_object
from transformers import AutoTokenizer, set_seed, get_cosine_schedule_with_warmup
from model.model import Transformer
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


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
    train_ds = PretrainDataset(data_paths, tokenizer, max_length=lm_config.max_seq_len)
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
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_loader) * args.epochs // args.accumulation_steps
    )
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    if args.load_dir is not None:
        accelerator.print(f'loading model from {args.load_dir}')
        accelerator.load_state(args.load_dir)
    if args.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None
    start_time = time.time()
    update_step = 0
    total_update_steps = len(train_loader) * args.epochs // args.accumulation_steps
    accumulation_loss = 0
    model.train()
    for epoch in range(args.epochs):
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            X = X.to(accelerator.device)
            Y = Y.to(accelerator.device)
            loss_mask = loss_mask.to(accelerator.device)
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()
            accumulation_loss += loss.item()
            accelerator.backward(loss)
            if (step + 1) % args.accumulation_steps == 0:
                if update_step % args.log_interval == 0:
                    accumulation_loss = sum(gather_object([accumulation_loss])) / accelerator.num_processes
                    spend_time = time.time() - start_time
                    accelerator.print('Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                            epoch, args.epochs, update_step, total_update_steps,
                            accumulation_loss, optimizer.param_groups[-1]['lr'],
                            (spend_time / (update_step + 1) * total_update_steps - spend_time) // 60
                        )
                    )
                    if wandb is not None and accelerator.is_main_process:
                        wandb.log({"loss": accumulation_loss, "lr": optimizer.param_groups[-1]['lr'],
                                "epoch_Time": (spend_time / (update_step + 1) * total_update_steps - spend_time) // 60})
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                update_step += 1
                accumulation_loss = 0
                if (update_step % args.save_interval == 0 or update_step == total_update_steps) and accelerator.is_main_process:
                    accelerator.save_model(model, args.save_dir)


# PORTS=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
# accelerate launch --multi_gpu --num_processes 8 --main_process_port ${PORTS[0]} 0-pretrain.py --use_wandb
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="Weights & Biases project name")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--data_dir", type=str, default="./dataset/pretrain_data/", help="Path to training data")
    parser.add_argument("--load_dir", type=str, default=None, help="Path to loading model")
    parser.add_argument("--save_dir", type=str, default="./out/pretrain/", help="Path to saving model")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")

    args = parser.parse_args()
    lm_config = LMConfig()
    args.wandb_run_name = f"Transformer-Length_{lm_config.max_seq_len}-Layer_{lm_config.n_layers}-Dim_{lm_config.dim}"
    torch.manual_seed(1337)
    set_seed(1337)

    train(args=args, lm_config=lm_config)
