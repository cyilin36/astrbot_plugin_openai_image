"""生图插件暴露给 LLM 的工具（function calling）。

LLM 通过本工具调用 ImageService 完成文生图 / 图生图：

- ``action=generate``：文生图
- ``action=edit``：图生图编辑，源图片取自当前消息（没有则取当前消息引用的图片），
  LLM 不需要也无法传图片字节
- ``action=status``：查询当前会话的生图任务，仅在用户主动询问进度时调用

提交类动作（generate / edit）本身不等待生图：参数校验通过后把任务交给插件自己的
后台登记表（``image_tasks.ImageTaskRegistry``），本工具立即结束当前回合、不返回任何
文本，因此模型不会再多说一句「正在生成」。图片生成完成后由插件直接发到会话，
不会唤醒模型；模型只有在用户开口时才通过 ``action=status`` 查询进度。

设计红线：call() 永不向 LLM 抛异常；返回的 JSON 文本绝不包含 api_key。
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import field as _stdlib_field
from typing import Any

try:  # AstrBot 以包方式加载插件：__import__(path, fromlist=["main"])
    from .image_errors import ImageError
    from .image_tasks import ImageTaskRegistry
except ImportError:  # 直接以顶层模块加载时的兜底
    from image_errors import ImageError
    from image_tasks import ImageTaskRegistry

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


TOOL_NAME = "openai_image"

# 非预期异常统一文案：不回显异常内容，避免泄露 api_key / 内部细节。
_GENERIC_ERROR = "生图失败，请稍后重试；若持续失败请联系管理员查看插件日志"

TOOL_DESCRIPTION = (
    "调用 OpenAI 兼容生图服务生成或编辑图片。本工具是后台任务：提交后立即返回，"
    "图片在后台生成，完成后会自动发到当前会话。"
    "调用本工具时不要输出任何文字说明：用户不需要「正在生成」这类回复，"
    "提交后本轮直接结束；也不要声称图片已经画好。"
    "用户询问进度、是否画好、为什么还没出来时，用 action=status 查询当前会话的生图任务；"
    "除此之外不要主动查询或主动汇报。"
    "action=generate 为文生图，需要 prompt；"
    "action=edit 为图生图编辑，待编辑的图片取自当前用户消息里附带的图片；"
    "用户引用了一张图片时，插件会自动取引用消息里的图片。"
    "你无需也无法传递图片字节；只有当前消息和引用消息里都没有图片时，"
    "才提示用户把图片随消息一起发出。"
    "prompt 默认直接使用用户原话：原样传递用户对画面的描述，"
    "不删减、不改写、不扩写、不润色、不补风格词、不翻译成英文；"
    "只有用户原话完全没有描述画面内容（例如只发了图片、或只说「再来一张」）时，"
    "才可以结合上下文做最小补全，并保持简短。"
    "生成参数默认尺寸 1024x1024、画质 auto；用户明确要求改尺寸 / 画质 / "
    "张数 / 背景 / 输出格式 / 模型时，直接传入对应字段即可，无需让用户改插件配置。"
)

_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["generate", "edit", "status"],
            "description": (
                "generate=文生图；edit=图生图编辑（图片取当前消息，没有则取引用消息）；"
                "status=查询当前会话的生图任务（仅在用户询问进度时使用）"
            ),
        },
        "prompt": {
            "type": "string",
            "description": (
                "生图或编辑提示词，action=generate / edit 时必填。"
                "默认直接使用用户原话，原样传递："
                "不删减（包括「帮我」「请」这类措辞）、不改写、不扩写、不润色、"
                "不补风格词、不翻译成英文。"
                "例如用户说「帮我生成一张和服美女图」，就传「帮我生成一张和服美女图」。"
                "仅当用户原话完全没有描述画面内容时才可结合上下文做最小补全。"
            ),
        },
        "task_id": {
            "type": "string",
            "description": "仅 action=status 使用：查询指定任务；留空返回当前会话的最近任务",
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
            "description": (
                "生成完成后是否自动把图片发到当前会话，默认 true；"
                "设为 false 时只保存本地路径（可在 action=status 的结果里看到）"
            ),
        },
    },
    "required": ["action"],
}

# 提前构造字段常量：dataclass 默认值里不允许出现函数调用（RUF009）
# default_factory 用 deepcopy，保证每个实例拿到互不共享的 schema（浅拷贝会共享嵌套 properties）
_SERVICE_FIELD = _tool_field(default=None, exclude=True)
_REGISTRY_FIELD = _tool_field(default_factory=ImageTaskRegistry, exclude=True)
_CONTEXT_FIELD = _tool_field(default=None, exclude=True)
_PARAMETERS_FIELD = _tool_field(default_factory=lambda: copy.deepcopy(_TOOL_PARAMETERS))


@_dataclass
class OpenAIImageTool(FunctionTool):  # type: ignore[misc]
    """LLM 工具：后台生图 / 图生图，以及按需查询任务状态。"""

    service: Any = _SERVICE_FIELD
    # 由插件注入的实例，保证 start() 与查询看到同一份任务记录。
    registry: Any = _REGISTRY_FIELD
    # Star 的 Context，用于在后台把图片/错误直接发到会话（不经过 LLM）。
    star_context: Any = _CONTEXT_FIELD
    name: str = TOOL_NAME
    description: str = TOOL_DESCRIPTION
    parameters: dict = _PARAMETERS_FIELD
    # 刻意不使用 AstrBot 的 is_background_task：那套机制会在后台任务完成后再唤醒
    # 主 agent 收尾，等于让模型多回复一轮。这里由插件自己起后台任务、自己发图。
    is_background_task: bool = False

    async def call(self, context: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """执行工具调用。

        generate / edit：校验后把生图挂到后台任务，然后结束本回合且不产出任何文本。
        status：返回当前会话的任务列表，供模型回答用户对进度的询问。

        始终以 JSON 文本或空结果产出，不会向调用方抛异常。
        """

        action = str(kwargs.get("action") or "").strip().lower()
        if action == "status":
            yield _dumps(self._status_payload(context, kwargs.get("task_id")))
            return
        if action not in ("generate", "edit"):
            yield _dumps(
                {"success": False, "error": "action 必须是 generate、edit 或 status"}
            )
            return

        event = _event_of(context)
        service = self.service
        if service is None or event is None:
            yield _dumps(
                {
                    "success": False,
                    "error": "生图服务尚未初始化，请检查插件是否正常加载",
                }
            )
            return

        prompt = str(kwargs.get("prompt") or "")
        send_to_user = kwargs.get("send_to_user")
        if send_to_user is None:
            send_to_user = True
        else:
            send_to_user = bool(send_to_user)
        params = _generation_params(kwargs)

        try:
            if action == "generate":
                uid, opts = service.prepare_generate(event, prompt, **params)
                images: list[tuple[str, bytes]] = []
            else:
                uid, opts, images = await service.prepare_edit(event, prompt, **params)
        except ImageError as exc:
            yield _dumps({"success": False, "error": str(exc)})
            return
        except Exception:  # 绝不让异常冒泡给 LLM
            logger.exception("生图工具参数校验失败")
            yield _dumps({"success": False, "error": _GENERIC_ERROR})
            return

        registry = self._task_registry()
        session = _session_of(event)
        task = registry.new_task(
            session, action, opts.prompt, model=opts.model, size=opts.size
        )
        registry.start(
            task,
            self._run_background(
                task=task,
                action=action,
                uid=uid,
                opts=opts,
                images=images,
                send_to_user=send_to_user,
                session=session,
                event=event,
            ),
        )
        # 生图已排入后台：结束本回合且不产出文本，模型不会再回复「正在生成」。
        yield _silent_result(event)

    # ------------------------------------------------------------------ 后台执行

    async def _run_background(
        self,
        *,
        task: Any,
        action: str,
        uid: str,
        opts: Any,
        images: list[tuple[str, bytes]],
        send_to_user: bool,
        session: str,
        event: Any,
    ) -> None:
        """在后台完成生图、发图，并把结果回写到任务记录。"""

        service = self.service
        try:
            if action == "edit":
                paths, _result = await service.run_edit(uid, opts, images)
            else:
                paths, _result = await service.run_generate(uid, opts)
        except asyncio.CancelledError:
            task.fail("任务已取消（插件卸载或服务停止）")
            raise
        except ImageError as exc:
            message = str(exc)
            task.fail(message)
            await self._notify_failure(session, event, message)
            return
        except Exception:  # 后台失败只记日志并回写状态，绝不向上抛
            logger.exception("后台生图失败")
            task.fail(_GENERIC_ERROR)
            await self._notify_failure(session, event, _GENERIC_ERROR)
            return

        if send_to_user:
            task.sent_to_user = await self._send_images(session, event, paths)
        task.succeed(paths)

    def _task_registry(self) -> Any:
        """惰性补齐任务登记表：正常运行时由插件注入，单测/离线场景自行创建。"""

        registry = self.registry
        if registry is None:
            registry = self.registry = ImageTaskRegistry()
        return registry

    # ------------------------------------------------------------------ 状态查询

    def _status_payload(self, context: Any, raw_task_id: Any) -> dict[str, Any]:
        """组装 action=status 的返回值；只暴露给 LLM 需要的信息。"""

        event = _event_of(context)
        session = _session_of(event)
        registry = self.registry
        if registry is None:
            return {"success": True, "tasks": [], "note": "当前会话还没有生图任务。"}

        wanted = str(raw_task_id or "").strip()
        if wanted:
            task = registry.get(session, wanted)
            if task is None:
                return {
                    "success": False,
                    "error": f"未找到任务 {wanted}；可只传 action=status 查询当前会话的任务。",
                }
            tasks = [task]
        else:
            tasks = registry.list(session)

        if not tasks:
            return {"success": True, "tasks": [], "note": "当前会话还没有生图任务。"}

        payload: dict[str, Any] = {
            "success": True,
            "tasks": [task.to_payload() for task in tasks],
        }
        payload["note"] = _status_note(tasks[0])
        return payload

    # ------------------------------------------------------------------ 消息发送

    async def _notify_failure(self, session: str, event: Any, message: str) -> None:
        """后台失败时主动发一条文字提示：此时没有 LLM 回合可以代劳。"""

        chain = _message_chain()
        if chain is None:
            logger.warning("未找到 MessageChain，跳过发送生图失败提示")
            return
        await self._send_chain(session, event, chain.message(message))

    async def _send_images(
        self, session: str, event: Any, paths: list[str]
    ) -> bool:
        """只把图片发到会话（不附带任何文案）；失败只记日志。"""

        chain = _message_chain()
        if chain is None:
            logger.warning("未找到 MessageChain，跳过发送生图结果")
            return False
        for path in paths:
            chain = chain.file_image(path)
        return await self._send_chain(session, event, chain)

    async def _send_chain(self, session: str, event: Any, chain: Any) -> bool:
        """优先用插件 Context 主动发送（事件流水线已结束，event.send 不保证可用）。"""

        try:
            if self.star_context is not None and session:
                return bool(await self.star_context.send_message(session, chain))
            await event.send(chain)
        except Exception as exc:  # noqa: BLE001 - 发送失败只记日志
            logger.warning("发送生图结果失败：%s", exc)
            return False
        return True


def _generation_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """把工具参数整理成 service 的生成参数；非法张数按未指定处理。"""

    raw_n = kwargs.get("n")
    try:
        n = int(raw_n) if raw_n is not None else 0
    except (TypeError, ValueError):
        n = 0
    return {
        "model": str(kwargs.get("model") or ""),
        "size": str(kwargs.get("size") or ""),
        "quality": str(kwargs.get("quality") or ""),
        "n": n,
        "background": str(kwargs.get("background") or ""),
        "output_format": str(kwargs.get("output_format") or ""),
    }


def _event_of(context: Any) -> Any:
    """从工具上下文里取 AstrMessageEvent；上下文异常时返回 None，绝不向外抛。"""

    try:
        inner = getattr(context, "context", None)
        return getattr(inner, "event", None)
    except Exception:  # noqa: BLE001 - 上下文实现可能抛异常
        return None


def _session_of(event: Any) -> str:
    """取会话标识（unified_msg_origin）；取不到时退化为空串。"""

    if event is None:
        return ""
    return str(getattr(event, "unified_msg_origin", "") or "")


def _silent_result(event: Any) -> Any:
    """构造「本回合结束且不发消息」的结果对象。

    框架收到 MessageEventResult 会把它设为事件结果；链为空时不会发送任何内容，
    而工具产出 None 会让 agent loop 直接结束，因此模型不会再多说一句话。
    """

    make_result = getattr(event, "make_result", None)
    if callable(make_result):
        return make_result()
    return None


def _status_note(task: Any) -> str:
    """给模型的下一步提示：避免重复发图或反复轮询。"""

    status = getattr(task, "status", "")
    if status == "running":
        return "仍在生成中，完成后图片会自动发到当前会话；不要反复调用本工具轮询。"
    if status == "failed":
        return "这次生成失败，请把 error 简要转述给用户。"
    if getattr(task, "sent_to_user", False):
        return "图片已经发到当前会话，直接简短告知用户即可，不要重复发送。"
    return "图片已生成但还没发到会话；需要发送时用 send_message_to_user 传入 paths 里的本地路径。"


def _message_chain() -> Any:
    """构造 MessageChain；运行环境缺失该模块时返回 None。"""

    try:
        from astrbot.core.message.message_event_result import MessageChain
    except ImportError:  # pragma: no cover - 取决于 astrbot 版本
        try:
            from astrbot.api.event import MessageChain
        except ImportError:  # pragma: no cover - 离线环境兜底
            return None
    return MessageChain()


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
