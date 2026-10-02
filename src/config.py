"""集中配置。

所有可调项都从这里取，不允许散落到各模块里硬编码——
否则后面做实验扫描（换模型、换档位、开关缓存）会改得到处都是。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class ModelSpec:
    """一个可调用的模型端点。

    `name` 是传给 litellm 的模型串，例如 `ollama_chat/qwen3:4b`。
    本地用 `ollama_chat/` 前缀（litellm 官方推荐，比 `ollama/` 响应更好）。
    """

    name: str
    tier: str  # "local" | "cloud"
    api_base: str | None = None
    api_key_env: str | None = None
    label: str = ""

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)


@dataclass(frozen=True)
class Config:
    local: ModelSpec
    cloud: ModelSpec

    # 采样与上下文：实验可复现性的关键，全部固定
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 2048
    # 默认值必须与 load() 里的 env 默认值一致，否则绕过 load() 直接构造时会
    # 悄悄拿到一个实测会掉 6.6 倍速的窗口（见 .env 里的拐点实测注释）。
    local_num_ctx: int = 16384
    # thinking 开关。本机实测同一道 GSM8K 题 think=ON 输出 1176 token /
    # 19.5s，think=OFF 仅 76 token / 3.2s——差 15 倍。必须作为受控变量固定。
    local_think: bool = False

    # 云端不可用时是否把 tier=cloud 的调用降级到本地模型。
    # 打开时：调用真的在本地跑，因此 tier 如实记 "local"，绝不虚报 cloud_tokens；
    # 被降级的事实另记在 requested_tier 字段里，报告时能算出「降级率」。
    # 关闭时：直接返回 error 结果（指标口径最纯净，但无 key 期间 dot 模式不可用）。
    cloud_fallback_local: bool = True

    request_timeout: int = 300
    db_path: Path = ROOT / "runs" / "metrics.db"
    cache_dir: Path = ROOT / "runs" / "cache"

    @staticmethod
    def load() -> "Config":
        local_model = ModelSpec(
            name=_env("LOCAL_MODEL", "ollama_chat/nanbeige4.2-3b-32k"),
            tier="local",
            api_base=_env("OLLAMA_BASE", "http://localhost:11434"),
            label="本地 SLM",
        )
        cloud_model = ModelSpec(
            name=_env("CLOUD_MODEL", "deepseek/deepseek-chat"),
            tier="cloud",
            api_key_env="DEEPSEEK_API_KEY",
            label="云端大模型",
        )
        return Config(
            local=local_model,
            cloud=cloud_model,
            temperature=float(_env("TEMPERATURE", "0.0")),
            seed=int(_env("SEED", "42")),
            max_tokens=int(_env("MAX_TOKENS", "2048")),
            local_num_ctx=int(_env("LOCAL_NUM_CTX", "16384")),
            local_think=_env("LOCAL_THINK", "0") == "1",
            cloud_fallback_local=_env("CLOUD_FALLBACK_LOCAL", "1") == "1",
        )


CONFIG = Config.load()

# 留出一截给模型输出和模板开销：窗口被输入填满时，输出会被静默截断
# （finish_reason=length，且不报错），是最难发现的一类指标污染。
OUTPUT_RESERVE = 1536


def cloud_ready() -> bool:
    """云端链路是否可用。未配置 key 时系统应能降级为纯本地运行。"""
    return bool(CONFIG.cloud.api_key)


def set_db_path(path: Path | str) -> Path:
    """重定向指标库（供 CLI 的 --db 使用）。

    这是**唯一受支持**的改库方式。不要从外部对 CONFIG 用 object.__setattr__，
    也不要各模块自己缓存路径——llm_client 落库走的是 CONFIG.db_path 这个单例，
    一处不同步就会让 calls 表和 tasks 表分居两库，
    按 task_id 汇总云端 token 恒为 0，且不报任何错。
    """
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    object.__setattr__(CONFIG, "db_path", p)  # CONFIG 是 frozen，仅此处开一个口子
    return p


def local_prompt_budget() -> int:
    """本地调用允许占用的 prompt token 上限。

    runner 用它决定「前置答案带多少、从何时开始丢最早的」。
    注意这不是字节数或字符数——runner 用的是历次调用回报的真实
    prompt_tokens / completion_tokens，所以这个预算是精确的，不用估。
    """
    return max(512, CONFIG.local_num_ctx - OUTPUT_RESERVE - CONFIG.max_tokens)
