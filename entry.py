"""
TinyZero QWen3 GRPO 训练入口。

自定义指标（reward 细分 + 采样日志）已直接集成到 verl 的 metric_utils.py 中。
"""

from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
