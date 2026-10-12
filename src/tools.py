"""子任务执行器可用的工具。

为什么需要它（实测动机，不是预设）：8B 模型在一条完全正确的分解链上，
把最后一步「25 − 22」算成了 1。分解对、推理链对，栽在一次一位数减法上。
**换更大的模型解决不了这个问题**——实测 3B 与 8B 的端到端准确率几乎无差别，
因为 LLM 的算术滑动在任何参数量上都会发生。

所以这里只做一件事：把算术从模型手里拿走，交给本地求值。工具在本地执行，
**不产生任何云端 token**，因此它同时是两个方向的改进——既提准确率，
又不增加指标一。

安全性：求值走 `ast` 白名单，只允许数字与四则/幂/取模运算。绝不使用 `eval`。
"""

from __future__ import annotations

import ast
import operator
from typing import Any

# 工具名。runner 在解析 tool_calls 时按它分派。
CALC_NAME = "calculator"

# 表达式长度上限。正常算式远短于此；超长基本是模型跑飞，直接拒绝而不是硬算。
MAX_EXPR_CHARS = 200

# 允许的二元运算符。**白名单**，不是黑名单。
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# 给模型的工具描述。措辞刻意强调「任何算术都先用它」——
# 实测模型对「简单的不必调工具」有强烈倾向，而那正是出错的地方。
CALCULATOR_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": CALC_NAME,
        "description": (
            "计算一个算术表达式并返回精确结果。"
            "**凡是涉及数字运算，都必须先用它算，不要自己心算**——"
            "哪怕是 25-22 这样的一步减法，也要交给它。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "纯算术表达式，例如 (25-14)*2 或 360*2/(6+4)",
                }
            },
            "required": ["expression"],
        },
    },
}


class CalculatorError(ValueError):
    """表达式不合法或不允许。"""


# 传给模型 API 的形态。**必须是 list**——Ollama 的 /api/chat 收到 dict 会直接
# 返回 HTTP 400，而且错误信息不含字段名，排查起来很费劲（踩过一次）。
CALCULATOR_TOOLS: list[dict[str, Any]] = [CALCULATOR_TOOL]


# 全角 → 半角 归一化表。中文输入法下模型经常输出全角符号，
# 不归一化就会白白失败一次（还要多占一轮对话）。
_FULLWIDTH = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "（": "(", "）": ")", "＋": "+", "－": "-", "×": "*",
    "÷": "/", "＊": "*", "／": "/", "　": " ",
})


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise CalculatorError(f"不支持的常量：{node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"不支持的运算符：{type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"不支持的一元运算符：{type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise CalculatorError(f"不支持的语法：{type(node).__name__}")


def calculate(expression: str) -> str:
    """求值一个算术表达式，返回结果的字符串形式。

    返回字符串而不是数字，因为它要直接作为 tool 消息的内容回灌给模型。
    失败时返回以「计算失败」开头的说明——**让模型看到失败原因**，
    比抛异常中断整条流水线有用得多（模型往往能据此改写表达式重试）。
    """
    expr = (expression or "").strip()
    if not expr:
        return "计算失败：表达式为空"
    if len(expr) > MAX_EXPR_CHARS:
        return f"计算失败：表达式过长（{len(expr)} 字符，上限 {MAX_EXPR_CHARS}）"
    # 全角符号在中文场景很常见，先归一化，免得模型写了个全角减号就失败。
    # 用 dict 形式而不是双字符串形式：后者要求两个串等长，数错一个字符就抛 ValueError。
    expr = expr.translate(_FULLWIDTH)
    try:
        value = _eval_node(ast.parse(expr, mode="eval"))
    except CalculatorError as e:
        return f"计算失败：{e}"
    except SyntaxError:
        return f"计算失败：不是合法的算术表达式（{expr[:60]}）"
    except ZeroDivisionError:
        return "计算失败：除数为零"
    except Exception as e:  # noqa: BLE001 - 求值再出意外也不能拖垮流水线
        return f"计算失败：{type(e).__name__}: {e}"

    # 整数结果不要显示成 3.0——下游汇聚看到浮点尾巴容易误判
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


if __name__ == "__main__":
    _samples = [
        "25-22", "(25-14)*2", "360*2/(6+4)", "12*8", "14*10-12*8",
        "45*3/5", "２５－２２", "1/0", "import os", "2**10", "abc",
    ]
    for s in _samples:
        print(f"  {s!r:22s} → {calculate(s)}")
