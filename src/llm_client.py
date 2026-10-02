"""唯一调用出口。

**系统里所有 LLM 调用都必须经过 call_llm()，没有例外。**

为什么这是一条硬规则：申请书要求"云端 token 累计所有 API 调用的输入与输出
（含规划、校验、重试）"。只要有一处调用绕过了这个函数，指标就是错的，而且
这种错不会抛异常、只会静默偏低——等发现时实验要全部重跑。

因此本模块的两条设计原则：
1. 落库发生在 call_llm 内部，调用方无法"忘记"记录。
2. 调用失败也要落库（error 字段），否则失败重试的 token 会被漏算。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# 必须在 import litellm 之前设：否则 litellm 会去 GitHub 拉远端价格表，
# 本机 SSL 证书链不全时每次 import 白等 4~8 秒（实测）。本地价格表足够用。
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402

from . import metrics  # noqa: E402
from .config import CONFIG, ModelSpec, cloud_ready  # noqa: E402

# litellm 默认会打一堆调试日志，压掉
litellm.suppress_debug_info = True
litellm.drop_params = True  # 模型不支持的参数自动丢弃，避免报错

# role 枚举。新增角色必须在此登记——否则指标里会冒出计划外的分组名，
# 「规划开销单独统计」这条口径就守不住了。
ROLES = ("decomposer", "executor", "aggregator", "warmup")

# strategy 枚举：dot = 子任务级分配（主方案）；direct = 整请求路由（对照基线）；
# warmup / smoke 仅供预热与冒烟，做实验统计时必须排除。
STRATEGIES = ("dot", "direct", "warmup", "smoke")


@dataclass
class LLMResult:
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    call_id: str = ""
    tier: str = ""  # 实际执行的档位（降级后 = "local"）
    requested_tier: str = ""  # 调用方要求的档位（未降级时与 tier 相同）
    model: str = ""
    usage_missing: bool = False
    error: str | None = None

    @property
    def fell_back(self) -> bool:
        """本次调用是否因云端不可用而被迫降级到本地。"""
        return bool(self.requested_tier) and self.requested_tier != self.tier


def _spec_for(tier: str) -> ModelSpec:
    return CONFIG.local if tier == "local" else CONFIG.cloud


def _extract_usage(resp: Any) -> tuple[int, int, int]:
    """从 litellm 响应里取 token 数。取不到时返回 (0,0,0)，由调用方告警。"""
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0, 0
    pt = getattr(u, "prompt_tokens", None) or 0
    ct = getattr(u, "completion_tokens", None) or 0
    cached = 0
    details = getattr(u, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None) or 0
    if not cached:
        # DeepSeek 用这个名字
        cached = getattr(u, "prompt_cache_hit_tokens", None) or 0
    return int(pt), int(ct), int(cached)


def _to_plain_tool_calls(msg: Any) -> list[dict[str, Any]]:
    tcs = getattr(msg, "tool_calls", None) or []
    out = []
    for tc in tcs:
        fn = getattr(tc, "function", None)
        out.append({
            "id": getattr(tc, "id", None) or f"call_{len(out)}",
            "name": getattr(fn, "name", None),
            "arguments": getattr(fn, "arguments", None) or "{}",
        })
    return out


def call_llm(
    *,
    messages: Sequence[dict[str, Any]],
    role: str,
    tier: str = "local",
    strategy: str = "dot",
    task_id: str = "",
    run_id: str = "default",
    tools: list[dict[str, Any]] | None = None,
    stream: bool = False,
    on_delta: Callable[[str], None] | None = None,
    retries: int = 0,
    **overrides: Any,
) -> LLMResult:
    """调用一次模型并落库。

    role    : decomposer | executor | aggregator | warmup —— 事后按角色归因 token。
              规划开销 = role IN ('decomposer','aggregator')，必须能单独算出来。
    tier    : local | cloud —— 决定是否计入云端 token 消耗。
              云端不可用时按 CONFIG.cloud_fallback_local 决定降级还是报错；
              降级后**实际档位**落在 result.tier，请求档位落在 result.requested_tier。
    strategy: dot | direct | warmup | smoke —— 本次实验的策略名，用于分组统计
    """
    if role not in ROLES:
        raise ValueError(f"未登记的 role={role!r}，请先加到 llm_client.ROLES：{ROLES}")

    # ① 先定实际档位：云端不可用且开了降级开关时，改走本地。
    requested_tier = tier
    effective_tier = tier
    if tier == "cloud" and not cloud_ready():
        if CONFIG.cloud_fallback_local:
            effective_tier = "local"
        else:
            result = LLMResult(
                tier="cloud",
                requested_tier=requested_tier,
                model=CONFIG.cloud.name,
                error="云端 API key 未配置（DEEPSEEK_API_KEY 为空）",
            )
            _persist(result, role=role, strategy=strategy, task_id=task_id, run_id=run_id)
            return result

    spec = _spec_for(effective_tier)
    kwargs: dict[str, Any] = {
        "model": spec.name,
        "messages": list(messages),
        "temperature": CONFIG.temperature,
        "seed": CONFIG.seed,
        "max_tokens": CONFIG.max_tokens,
        "timeout": CONFIG.request_timeout,
    }
    if spec.api_base:
        kwargs["api_base"] = spec.api_base
    if spec.api_key:
        kwargs["api_key"] = spec.api_key
    if tools:
        kwargs["tools"] = tools

    # 本地模型的额外控制：上下文长度与 thinking 开关。
    # 注意 num_ctx 不设的话 Ollama 默认只有 4096，长上下文实验会测不出效果。
    #
    # 必须走顶层 kwargs：litellm 的 ollama_chat 会把它们并进 body 的 options，
    # 且 drop_params 不会丢。**不要改成 extra_body**——实测 extra_body 里的
    # options 是整体替换而非合并，会把 temperature/seed/max_tokens 一起抹掉。
    # 验证脚本见 notes/_probe_litellm_body.py。
    if effective_tier == "local":
        kwargs["num_ctx"] = CONFIG.local_num_ctx
        kwargs["think"] = CONFIG.local_think

    # 流式下 usage 只在最后一个 chunk 出现；不显式要，多数端点就不给。
    # ⚠ stream=True 这一句是必须的：只加 stream_options 而不设 stream，
    # litellm 会当非流式请求发出去，返回的是一个 ModelResponse 对象；
    # 对它迭代拿到的是 tuple，_stream_call 里 chunk.choices 直接 AttributeError。
    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

    kwargs.update(overrides)

    # 每次尝试都单独落库：申请书口径是「累计所有 API 调用的输入与输出（含重试）」，
    # 所以失败那次的 prompt token 也算钱，合并成一行反而是错的。
    attempts = max(1, retries + 1)
    result = LLMResult(tier=effective_tier, requested_tier=requested_tier, model=spec.name)

    for attempt in range(1, attempts + 1):
        t0 = time.perf_counter()
        try:
            if stream:
                _stream_call(kwargs, result, on_delta, t0)
            else:
                _oneline_call(kwargs, result, t0)
            result.error = None
        except KeyboardInterrupt:
            # KeyboardInterrupt 继承 BaseException，不会被下面的 except Exception 接住。
            # 而它恰好是「token 已经花掉、但结果被丢弃」的典型场景（用户流式看到一半按 Ctrl+C）。
            # 不落库的话这笔消耗就静默消失了，违反「每次调用无条件落库」这条红线。
            result.latency_ms = (time.perf_counter() - t0) * 1000
            result.error = "KeyboardInterrupt: 用户中断"
            _persist(result, role=role, strategy=strategy, task_id=task_id, run_id=run_id)
            raise  # 记完账再把中断抛给上层，不能吞掉，否则 Ctrl+C 会失灵
        except Exception as e:  # noqa: BLE001
            result.latency_ms = (time.perf_counter() - t0) * 1000
            result.error = f"{type(e).__name__}: {e}"

        _persist(result, role=role, strategy=strategy, task_id=task_id, run_id=run_id)

        if result.error is None or attempt >= attempts:
            break
        # 流式已经吐过字就不能重试，否则用户会看到重复内容
        if stream and (result.text or result.reasoning):
            break
        if not _is_transient(result.error):
            break
        time.sleep(min(2.0 ** (attempt - 1), 8.0))
        result = LLMResult(tier=effective_tier, requested_tier=requested_tier, model=spec.name)

    return result


# 本机实测：RTX 5060 Laptop（sm_120）冷加载偶发
# "CUDA error: shared object initialization failed" / 0xc0000409，
# Ollama 自己第二次加载就成功了（server.log 可见 Load failed → 紧随其后成功）。
# 这类错误重试一次几乎必过，但不能算「成功」——失败的尝试已按上面的规则单独落库。
_TRANSIENT_MARKERS = (
    "shared object initialization failed",
    "process has terminated",
    "CUDA error",
    "InternalServerError",
    "APIConnectionError",
    "Timeout",
    "timed out",
    "connection reset",
)


def _is_transient(err: str) -> bool:
    low = err.lower()
    return any(m.lower() in low for m in _TRANSIENT_MARKERS)


def _oneline_call(kwargs: dict[str, Any], result: LLMResult, t0: float) -> None:
    resp = litellm.completion(**kwargs)
    result.latency_ms = (time.perf_counter() - t0) * 1000
    msg = resp.choices[0].message
    result.text = getattr(msg, "content", "") or ""
    result.reasoning = getattr(msg, "reasoning_content", None) or ""
    result.tool_calls = _to_plain_tool_calls(msg)
    pt, ct, cached = _extract_usage(resp)
    result.prompt_tokens, result.completion_tokens, result.cache_hit_tokens = pt, ct, cached
    result.usage_missing = (pt == 0 and ct == 0)


def _stream_call(
    kwargs: dict[str, Any],
    result: LLMResult,
    on_delta: Callable[[str], None] | None,
    t0: float,
) -> None:
    """流式调用。

    注意 1：流式下 thinking 内容走独立的 reasoning 字段，content 可能长期为空。
    只累加 content 会严重漏算 token，所以 token 一律以 usage 为准，不自己数。

    注意 2：这里**逐片写回 result.text / result.reasoning**，而不是先攒 list
    最后再 join。因为 call_llm 的重试守卫要判断「是否已经吐过字」——
    若只在循环结束后才赋值，中途异常时 result.text 仍是空的，守卫永远不触发，
    重试会把已经流出去的内容整段重发（用户看到两遍，on_delta 也重复打印）。
    """
    for chunk in litellm.completion(**kwargs):
        if result.ttft_ms is None:
            delta_probe = chunk.choices[0].delta if chunk.choices else None
            if delta_probe is not None and (
                getattr(delta_probe, "content", None) or getattr(delta_probe, "reasoning_content", None)
            ):
                result.ttft_ms = (time.perf_counter() - t0) * 1000
        choices = getattr(chunk, "choices", None) or []
        if choices:
            delta = choices[0].delta
            piece = getattr(delta, "content", None) or ""
            if piece:
                result.text += piece  # 逐片写回，见 docstring 注意 2
                if on_delta:
                    on_delta(piece)
            rz = getattr(delta, "reasoning_content", None) or ""
            if rz:
                result.reasoning += rz
            tcs = _to_plain_tool_calls(delta)
            if tcs:
                result.tool_calls.extend(tcs)
        # usage 只在最后一个 chunk 出现
        if getattr(chunk, "usage", None):
            pt, ct, cached = _extract_usage(chunk)
            if pt or ct:
                result.prompt_tokens, result.completion_tokens, result.cache_hit_tokens = pt, ct, cached
    result.latency_ms = (time.perf_counter() - t0) * 1000
    result.usage_missing = (result.prompt_tokens == 0 and result.completion_tokens == 0)


def _persist(result: LLMResult, **kw: Any) -> None:
    if result.usage_missing and not result.error:
        # 这不是小事：token 全 0 会让三指标直接失真，必须吵出来
        print(
            f"[llm_client] ⚠ {result.model} 未返回 usage，token 记为 0。"
            f" 检查该模型/接口是否支持 usage 字段，否则指标不可用。"
        )
    rec = metrics.record(
        metrics.CallRecord(
            role=kw["role"],
            tier=result.tier,
            requested_tier=result.requested_tier or result.tier,
            model=result.model,
            strategy=kw["strategy"],
            task_id=kw["task_id"],
            run_id=kw["run_id"],
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cache_hit_tokens=result.cache_hit_tokens,
            latency_ms=result.latency_ms,
            ttft_ms=result.ttft_ms,
            error=result.error,
        )
    )
    result.call_id = rec.call_id


def warmup(tier: str = "local", rounds: int = 1) -> float:
    """预热模型，把加载时间从测量里剔除。

    本机实测：冷启动首调 37.34s（其中含模型加载），预热后同规模 prompt 仅 0.04s。
    不做 warmup，P95 完全失真。

    这里带重试不是"保险起见"：本机 RTX 5060 Laptop（sm_120）冷加载会偶发
    CUDA 初始化失败，Ollama 自己重试才成功。预热正好把这个坑吃掉，
    所以预热阶段必须容忍重试——否则崩溃会被算进正式测量。
    """
    t0 = time.perf_counter()
    for i in range(rounds):
        r = call_llm(
            messages=[{"role": "user", "content": "hi"}],
            role="warmup",
            tier=tier,
            strategy="warmup",
            task_id="__warmup__",
            retries=3,
            max_tokens=1,
        )
        if r.error:
            print(f"[llm_client] ⚠ warmup({tier}) 第 {i + 1} 轮仍失败：{r.error[:160]}")
    return time.perf_counter() - t0


if __name__ == "__main__":
    dt = warmup()
    print(f"warmup 用时 {dt:.2f}s")
    r = call_llm(
        messages=[{"role": "user", "content": "用一句话说明什么是端云协同。"}],
        role="executor",
        tier="local",
        strategy="smoke",
        task_id="smoke",
        retries=2,
    )
    print(f"tier={r.tier} model={r.model}")
    print(f"tokens: prompt={r.prompt_tokens} completion={r.completion_tokens}")
    print(f"latency={r.latency_ms:.0f}ms error={r.error}")
    print(f"text: {r.text[:200]}")
