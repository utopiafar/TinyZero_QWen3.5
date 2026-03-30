"""
TinyZero QWen3.5 GRPO 训练入口。

通过 monkey-patching 将自定义 reward 函数注入 verl 框架，
无需修改 verl 源码。

工作原理:
    1. 导入 verl 的 main_ppo 模块
    2. 替换 _select_rm_score_fn，添加 countdown 支持
    3. 调用 verl 原始的 main() 函数启动训练

用法:
    python entry.py data.train_files=... data.val_files=... [其他 hydra 参数]
    # 或
    bash scripts/train_grpo.sh  # (内部调用 entry.py)
"""

import sys
from src.reward.countdown import compute_score as countdown_compute_score


def _patch_reward_function():
    """将 countdown reward 函数注入 verl 的 reward 分发机制。"""
    try:
        import verl.trainer.main_ppo as main_ppo_module

        # 保存原始函数
        if hasattr(main_ppo_module, '_select_rm_score_fn'):
            _original_select_fn = main_ppo_module._select_rm_score_fn
        else:
            _original_select_fn = None

        def _patched_select_fn(data_source):
            """扩展版 reward 分发函数，支持 countdown 任务。"""
            if "countdown" in data_source:
                return countdown_compute_score

            # 回退到原始分发逻辑
            if _original_select_fn is not None:
                return _original_select_fn(data_source)

            raise NotImplementedError(
                f"Unknown data_source: {data_source}. "
                f"仅支持 'countdown' 任务。"
            )

        # 替换模块级函数
        main_ppo_module._select_rm_score_fn = _patched_select_fn
        print("[entry.py] 已注入 countdown reward 函数到 verl")

    except ImportError as e:
        print(f"[entry.py] 警告: 无法导入 verl.trainer.main_ppo: {e}")
        print("[entry.py] 请确保 verl 已安装: pip install verl==0.7.1")
        sys.exit(1)


def main():
    """主入口函数。"""
    _patch_reward_function()

    # 导入 verl 的 main 函数 (hydra 入口)
    from verl.trainer.main_ppo import main as verl_main

    print("[entry.py] 启动 verl GRPO 训练...")
    verl_main()


if __name__ == '__main__':
    main()
