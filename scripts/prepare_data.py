"""
Countdown 任务数据预处理脚本。

从 HuggingFace 加载 Jiayi-Pan/Countdown-Tasks-3to4 数据集，
转换为 verl 框架所需的 Parquet 格式。

使用方法:
    python scripts/prepare_data.py --local_dir ~/data/countdown
    python scripts/prepare_data.py --local_dir /root/autodl-tmp/data/countdown --train_size 327680
"""

import os
import argparse
from datasets import load_dataset
from src.templates.countdown import make_prefix

DATA_SOURCE = 'countdown'


def make_map_fn(split, template_fn):
    """创建数据集映射函数。"""
    def process_fn(example, idx):
        question = template_fn(example)

        solution = {
            "target": example['target'],
            "numbers": example['nums'],
        }

        return {
            "data_source": DATA_SOURCE,
            "prompt": [{
                "role": "user",
                "content": question,
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
    parser.add_argument('--local_dir', default='~/data/countdown',
                        help='本地输出目录')
    parser.add_argument('--train_size', type=int, default=327680,
                        help='训练集大小')
    parser.add_argument('--test_size', type=int, default=1024,
                        help='测试集大小')
    args = parser.parse_args()

    # 加载原始数据集
    print(f"从 HuggingFace 加载 Countdown 数据集...")
    raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
    print(f"原始数据集大小: {len(raw_dataset)}")

    # 验证数据集足够大
    total_needed = args.train_size + args.test_size
    assert len(raw_dataset) > total_needed, \
        f"数据集太小: 需要 {total_needed}, 实际 {len(raw_dataset)}"

    # 划分训练集和测试集
    train_dataset = raw_dataset.select(range(args.train_size))
    test_dataset = raw_dataset.select(range(args.train_size, args.train_size + args.test_size))

    # 使用 Qwen3.5 模板处理数据
    template_fn = make_prefix

    print(f"处理训练集 ({args.train_size} 样本)...")
    train_dataset = train_dataset.map(function=make_map_fn('train', template_fn), with_indices=True)

    print(f"处理测试集 ({args.test_size} 样本)...")
    test_dataset = test_dataset.map(function=make_map_fn('test', template_fn), with_indices=True)

    # 保存为 Parquet 格式
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_path = os.path.join(local_dir, 'train.parquet')
    test_path = os.path.join(local_dir, 'test.parquet')

    print(f"保存训练集到: {train_path}")
    train_dataset.to_parquet(train_path)

    print(f"保存测试集到: {test_path}")
    test_dataset.to_parquet(test_path)

    print(f"完成! 训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")


if __name__ == '__main__':
    main()
