"""
Countdown 任务数据预处理脚本。

使用 apply_chat_template + enable_thinking=True 生成 prompt（参照 CoLA-RL 教程）。
通过 reward.custom_reward_function.path 注入自定义 reward 函数（verl v0.7.1 方式）。

使用方法:
    python scripts/prepare_data.py --local_dir /root/autodl-tmp/data/countdown
"""

import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

DATA_SOURCE = 'countdown'
DEFAULT_MODEL_PATH = '/root/autodl-fs/models/Qwen3-1.7B-Base'

PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your reasoning in <think reasoning> </think reasoning> tags, "
    "and return the final answer in <answer> </answer> tags, "
    "e.g. <answer> (1 + 2) / 3 </answer>."
)


def make_map_fn(split, tokenizer):
    def process_fn(example, idx):
        numbers = example['nums']
        target = example['target']

        user_content = PROMPT_TEMPLATE.format(numbers=numbers, target=target)
        messages = [{"role": "user", "content": user_content}]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        solution = {
            "target": target,
            "numbers": numbers,
        }

        return {
            "data_source": DATA_SOURCE,
            "prompt": [{
                "content": prompt_text,
                "role": "user",
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                'split': split,
                'index': idx,
            },
        }
    return process_fn


def main():
    parser = argparse.ArgumentParser(description='生成 Countdown 训练数据')
    parser.add_argument('--local_dir', default='/root/autodl-tmp/data/countdown')
    parser.add_argument('--train_size', type=int, default=32768)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    print(f"从 {args.model_path} 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"从 HuggingFace 加载 Countdown 数据集...")
    raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
    print(f"原始数据集大小: {len(raw_dataset)}")

    total_needed = args.train_size + args.test_size
    assert len(raw_dataset) > total_needed

    train_dataset = raw_dataset.select(range(args.train_size))
    test_dataset = raw_dataset.select(range(args.train_size, args.train_size + args.test_size))

    print(f"处理训练集 ({args.train_size})...")
    train_dataset = train_dataset.map(function=make_map_fn('train', tokenizer), with_indices=True)

    print(f"处理测试集 ({args.test_size})...")
    test_dataset = test_dataset.map(function=make_map_fn('test', tokenizer), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_path = os.path.join(local_dir, 'train.parquet')
    test_path = os.path.join(local_dir, 'test.parquet')

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    sample = train_dataset[0]
    print(f"\n样本 prompt (前200字): {sample['prompt'][0]['content'][:200]}...")
    print(f"ground_truth: {sample['reward_model']['ground_truth']}")
    print(f"\n完成! 训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")


if __name__ == '__main__':
    main()
