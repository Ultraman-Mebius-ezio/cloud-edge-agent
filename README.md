# cloud-edge-agent

端云协同智能体原型。把一个任务拆成子任务，**逐个**判断交给本地小模型还是云端大模型，
执行后汇总。研究点是**子任务级分配**，相对「整请求路由」的改进所在。

## 快速开始

```bash
uv venv .venv
uv pip install litellm python-dotenv rich
cp .env.example .env          # 填 DEEPSEEK_API_KEY
ollama pull nanbeige4.2-3b-32k

# Windows 必须加 PYTHONIOENCODING，否则控制台中文乱码
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m src.cli --mode dot --once "一个班 45 人，其中 3/5 是女生，女生比男生多几人？" --run-id exp1
```

没配云端 key 也能跑：`tier=cloud` 的调用会落到本地执行，并**如实记为 local**，
不会虚报云端消耗。

## 两种模式

| 模式 | 行为 | 用途 |
|---|---|---|
| `--mode dot` | 分解 → 逐子任务分配 → 顺序执行 → 汇聚 | 主方案（默认） |
| `--mode direct` | 整请求判定走本地或云端，一次答完 | 对照基线（DataShunt 式） |

交互对话去掉 `--once` 即可。指标看板：

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m src.cli --stats --run-id exp1
```

三张表：请求级对照（p50/p95、云端 token、SLM 使用率）、规划开销 vs 执行开销、
按 role×tier 的调用明细。

## 指标口径

| 指标 | 定义 |
|---|---|
| 云端 token 消耗 | Σ(`tier=cloud` 的 prompt+completion)，**含分解、汇聚、失败与重试的每一次尝试** |
| 完整响应时间 | 请求进 → 结果出，含本地推理。**请求级**量纲，报中位数 + P95 |
| SLM 使用率 | 走 local 的子任务数 / 总子任务数。分配器判定与实际执行分列两栏 |

规划开销（`role IN ('decomposer','aggregator')`）单独统计。DoT 论文没有计量这笔开销，
正是本项目要报告的点。

## 项目结构

```
src/config.py        配置（含 num_ctx 实测拐点）
src/metrics.py       SQLite 落库与汇总（calls 表 + tasks 表）
src/llm_client.py    ★ 唯一调用出口 + warmup + 云端降级
src/planner.py       分解（云端）
src/allocator.py     难度分配（纯规则，零 LLM 调用）
src/runner.py        顺序执行（本地/云端混合）
src/aggregator.py    汇聚（云端）
src/cli.py           入口、--mode、--stats 看板
tests/smoke_test.py  冒烟测试（不依赖 pytest）
```

**所有 LLM 调用必须走 `llm_client.call_llm()`，没有例外。** 落库在该函数内部完成，
调用方无法「忘记」记录——绕过会导致云端 token 静默偏低且不报错。

## 测试

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tests.smoke_test             # 含端到端，约 10s
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tests.smoke_test --skip-e2e   # 只跑纯函数
```

## 文档

- `docs/开发说明.md` —— 完整说明：实测结论、固定项、环境坑、开发约定
- `docs/接口冻结.md` —— 模块契约，改接口前先读
- `docs/工作日志.md` —— 进展记录

## 已知限制

- 本机 8GB 显存，本地模型固定 3B；DoT 论文用 8B，差距需在报告中说明
- v4 不做依赖图与并行执行，拆解的收益只剩「把简单子任务下沉到本地省 token」
- 难度分配用阈值规则，不是论文里的训练 adapter；论文中不训练时 P3 仅 15%
