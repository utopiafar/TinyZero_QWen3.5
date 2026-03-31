"""
Countdown 任务数据预处理脚本。

手拼 Qwen3 ChatML 模板生成 prompt，无需 transformers>=5.2 的 enable_thinking。

使用方法:
    python scripts/prepare_data.py --local_dir /root/autodl-tmp/data/countdown
"""

import os
import random
import argparse
from datasets import load_dataset, Dataset

DATA_SOURCE = 'countdown'

PROMPT_TEMPLATE = (
    "<|im_start|>user\n"
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your reasoning in <think reasoning> </think reasoning> tags, "
    "and return the final answer in <answer> </answer> tags, "
    "e.g. <answer> (1 + 2) / 3 </answer>.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def generate_countdown_data(n_samples, n_nums=4, seed=42):
    """生成 Countdown 数据：随机选 3-4 个数字(1-100)，随机选 target(1-100)。"""
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        count = rng.choice([3, 4]) if n_nums == 4 else n_nums
        numbers = [rng.randint(1, 100) for _ in range(count)]
        target = rng.randint(1, 100)
        data.append({"nums": numbers, "target": target})
    return data


def make_map_fn(split):
    def process_fn(example, idx):
        numbers = example['nums']
        target = example['target']

        prompt_text = PROMPT_TEMPLATE.format(numbers=numbers, target=target)

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
    args = parser.parse_args()

    print("从 HuggingFace 加载 Countdown 数据集...")
    try:
        raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
        print(f"原始数据集大小: {len(raw_dataset)}")
        total_needed = args.train_size + args.test_size
        assert len(raw_dataset) > total_needed
        train_raw = raw_dataset.select(range(args.train_size))
        test_raw = raw_dataset.select(range(args.train_size, args.train_size + args.test_size))
    except Exception as e:
        print(f"网络不可用 ({e})，使用合成数据...")
        train_raw = Dataset.from_list(generate_countdown_data(args.train_size, seed=42))
        test_raw = Dataset.from_list(generate_countdown_data(args.test_size, seed=123))

    print(f"处理训练集 ({len(train_raw)})...")
    train_dataset = train_raw.map(function=make_map_fn('train'), with_indices=True)

    print(f"处理测试集 ({len(test_raw)})...")
    test_dataset = test_raw.map(function=make_map_fn('test'), with_indices=True)

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
