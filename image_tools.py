"""生图插件暴露给 LLM 的工具（function calling）。

LLM 通过本工具调用 ImageService 完成文生图 / 图生图：
- action=generate：文生图
- action=edit：图生图编辑，源图片取自当前消息，LLM 不需要也无法传图片字节

设计红线：call() 永不向 LLM 抛异常；返回的 JSON 文本绝不包含 api_key。
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import field as _stdlib_field
from typing import Any

try:  # AstrBot 以包方式加载插件：__import__(path, fromlist=["main"])
    from .image_errors import ImageError
except ImportError:  # 直接以顶层模块加载时的兜底
    from image_errors import ImageError

try:  # pragma: no cover - 取决于运行环境
    from astrbot.api import logger as _astrbot_logger
except Exception:  # noqa: BLE001 - astrbot 缺失时回退标准库日志
    _astrbot_logger = None

logger: Any = (
    _astrbot_logger if _astrbot_logger is not None else logging.getLogger(__name__)
)

# AstrBot 的 FunctionTool 是 pydantic dataclass（astrbot/core/agent/tool.py 用
# pydantic.dataclasses.dataclass 定义），因此 service 字段的 exclude=True 需要
# pydantic 的 Field。缺失 pydantic 时退回标准库 dataclass 并忽略 exclude。
try:  # pragma: no cover - 取决于运行环境
    from pydantic import Field as _PydanticField
    from pydantic.dataclasses import dataclass as _dataclass
except ImportError:  # pragma: no cover - 无 pydantic 的离线兜底
    _PydanticField = None

    from dataclasses import dataclass as _dataclass


def _tool_field(
    *, default: Any = None, exclude: bool = False, default_factory: Any = None
) -> Any:
    """构造工具字段：优先 pydantic Field，无 pydantic 时退回标准库 field。"""

    if _PydanticField is not None:
        kwargs: dict[str, Any] = {"exclude": exclude}
        if default_factory is not None:
            return _PydanticField(default_factory=default_factory, **kwargs)
        return _PydanticField(default=default, **kwargs)
    if default_factory is not None:
        return _stdlib_field(default_factory=default_factory)
    return _stdlib_field(default=default)


try:  # pragma: no cover - 取决于 astrbot 版本
    from astrbot.api import FunctionTool
except ImportError:  # pragma: no cover
    try:
        from astrbot.core.agent.tool import FunctionTool
    except ImportError:  # pragma: no cover - 离线环境兜底

        class FunctionTool:  # type: ignore[no-redef]
            """离线兜底占位：astrbot 不可用时仍可导入本模块。"""


try:  # pragma: no cover - 取决于 astrbot 版本
    from astrbot.core.agent.tool import ToolExecResult
except ImportError:  # pragma: no cover
    try:
        from astrbot.api import ToolExecResult
    except ImportError:  # pragma: no cover - 离线环境兜底
        ToolExecResult = str  # type: ignore[assignment,misc]

TOOL_NAME = "openai_image"

TOOL_DESCRIPTION = (
    "调用 OpenAI 兼容生图服务生成或编辑图片。"
    "action=generate 为文生图，仅需 prompt；"
    "action=edit 为图生图编辑，待编辑的图片取自当前用户消息里附带的图片，"
    "你无需也无法传递图片字节，此时应提示用户把图片随消息一起发出。"
    "prompt 默认直接使用用户原话：原样传递用户对画面的描述，"
    "不删减、不改写、不扩写、不润色、不补风格词、不翻译成英文；"
    "只有用户原话完全没有描述画面内容（例如只发了图片、或只说「再来一张」）时，"
    "才可以结合上下文做最小补全，并保持简短。"
    "返回保存后的本地图片路径，默认同时把图片发给用户。"
    "生成参数默认尺寸 1024x1024、画质 auto；用户明确要求改尺寸 / 画质 / "
    "张数 / 背景 / 输出格式 / 模型时，直接传入对应字段即可，无需让用户改插件配置。"
)

_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["generate", "edit"],
            "description": "generate=文生图；edit=图生图编辑（图片取当前消息）",
        },
        "prompt": {
            "type": "string",
            "description": (
                "生图或编辑提示词。默认直接使用用户原话，原样传递："
                "不删减（包括「帮我」「请」这类措辞）、不改写、不扩写、不润色、"
                "不补风格词、不翻译成英文。"
                "例如用户说「帮我生成一张和服美女图」，就传「帮我生成一张和服美女图」。"
                "仅当用户原话完全没有描述画面内容时才可结合上下文做最小补全。"
            ),
        },
        "size": {
            "type": "string",
            "description": (
                "图片尺寸，默认 1024x1024。可选 1536x1024 / 1024x1536 / auto，"
                "或自定义 宽x高（如 1280x720，两边为 16 的倍数、宽高比 1:3~3:1、"
                "最长边不超过 3840）；留空用插件默认值"
            ),
        },
        "quality": {
            "type": "string",
            "enum": ["auto", "low", "medium", "high", "xhigh", "max"],
            "description": (
                "画质，默认 auto（不指定时即 auto）：low / medium / high / "
                "xhigh / max（xhigh、max 仅部分模型支持）；留空用插件默认值"
            ),
        },
        "background": {
            "type": "string",
            "description": "背景，如 auto / transparent / opaque，留空用插件默认值",
        },
        "output_format": {
            "type": "string",
            "description": "输出格式，png / jpeg / webp，留空用插件默认值",
        },
        "n": {
            "type": "integer",
            "minimum": 1,
            "description": "生成张数，受插件 max_n 配置限制",
        },
        "model": {
            "type": "string",
            "description": "模型名，留空使用插件配置的默认模型",
        },
        "send_to_user": {
            "type": "boolean",
            "default": True,
            "description": "是否直接把生成结果发到当前会话，默认 true",
        },
    },
    "required": ["action", "prompt"],
}

# 提前构造字段常量：dataclass 默认值里不允许出现函数调用（RUF009）
# default_factory 用 deepcopy，保证每个实例拿到互不共享的 schema（浅拷贝会共享嵌套 properties）
_SERVICE_FIELD = _tool_field(default=None, exclude=True)
_PARAMETERS_FIELD = _tool_field(default_factory=lambda: copy.deepcopy(_TOOL_PARAMETERS))


@_dataclass
class OpenAIImageTool(FunctionTool):  # type: ignore[misc]
    """LLM 工具：文生图 / 图生图。"""

    service: Any = _SERVICE_FIELD
    name: str = TOOL_NAME
    description: str = TOOL_DESCRIPTION
    parameters: dict = _PARAMETERS_FIELD

    async def call(self, context: Any, **kwargs: Any) -> ToolExecResult:
        action = str(kwargs.get("action") or "").strip().lower()
        if action not in ("generate", "edit"):
            return _dumps({"success": False, "error": "action 必须是 generate 或 edit"})

        prompt = str(kwargs.get("prompt") or "")
        send_to_user = kwargs.get("send_to_user")
        if send_to_user is None:
            send_to_user = True
        else:
            send_to_user = bool(send_to_user)

        params = {
            "model": str(kwargs.get("model") or ""),
            "size": str(kwargs.get("size") or ""),
            "quality": str(kwargs.get("quality") or ""),
            "background": str(kwargs.get("background") or ""),
            "output_format": str(kwargs.get("output_format") or ""),
        }
        raw_n = kwargs.get("n")
        try:
            params["n"] = int(raw_n) if raw_n is not None else 0
        except (TypeError, ValueError):
            params["n"] = 0

        try:
            event = context.context.event
            service = self.service
            if service is None:
                return _dumps(
                    {
                        "success": False,
                        "error": "生图服务尚未初始化，请检查插件是否正常加载",
                    }
                )
            if action == "generate":
                paths, result = await service.generate(event, prompt, **params)
            else:
                paths, result = await service.edit(event, prompt, **params)
        except ImageError as exc:
            return _dumps({"success": False, "error": str(exc)})
        except Exception:  # 绝不让异常冒泡给 LLM
            logger.exception("生图工具执行失败")
            return _dumps(
                {
                    "success": False,
                    "error": "生图失败，请稍后重试；若持续失败请联系管理员查看插件日志",
                }
            )

        model = _text(getattr(result, "model", "")) or _text(
            getattr(service, "_last_model", "")
        )
        size = _text(getattr(service, "_last_size", ""))

        payload: dict[str, Any] = {
            "success": True,
            "count": len(paths),
            "paths": list(paths),
            "model": model,
            "size": size,
            "sent_to_user": False,
        }

        if send_to_user:
            sent = await self._send(event, paths)
            payload["sent_to_user"] = sent
            if not sent:
                payload["warning"] = "图片已生成但发送失败，可查看插件日志"

        return _dumps(payload)

    async def _send(self, event: Any, paths: list[str]) -> bool:
        """只把图片发到当前会话（不附带任何文案）；失败只记日志，不影响返回值。"""

        try:
            from astrbot.core.message.message_event_result import MessageChain
        except ImportError:  # pragma: no cover
            try:
                from astrbot.api.event import MessageChain
            except ImportError:  # pragma: no cover - 离线环境兜底
                logger.warning("未找到 MessageChain，跳过主动发送图片")
                return False
        try:
            chain = MessageChain()
            for path in paths:
                chain = chain.file_image(path)
            await event.send(chain)
        except Exception as exc:  # noqa: BLE001 - 发送失败只记日志
            logger.warning("发送生图结果失败：%s", exc)
            return False
        return True


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)
