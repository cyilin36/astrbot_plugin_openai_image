"""AstrBot OpenAI 生图插件入口。

只暴露一条顶层指令 ``/draw``（别名 ``/生图``），子能力由内部手动路由：

- ``/draw <提示词>``         文生图
- ``/draw edit <提示词>``    图生图（图片随同一条消息发出）
- ``/draw help``             使用说明

路由刻意不使用「指令组 + 组内子指令」：AstrBot 4.28.2 的 waking 阶段会把**所有**
filter 命中的 handler 一起激活（``pipeline/waking_check/stage.py``），而指令组子指令的
完整名由父名拼接而成（``"draw edit"`` 等），一旦同时存在顶层同名指令，同一条消息就会
被两个 handler 同时处理。实测「顶层裸指令 draw + 同名指令组内 edit 子指令」对
``/draw edit x`` 会激活 2 个 handler，即重复生图、重复扣费。因此这里只用一条顶层指令、
内部手动路由，实测每条消息恒为 1 个激活 handler。

同理，``args`` 必须声明为 ``GreedyStr`` 且**不能带默认值**：``init_handler_md``
会把带默认值的参数记录成默认值本身，使 ``is_greedy`` 判定失败，提示词被截断
成首个 token。
"""

from __future__ import annotations

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.star.filter.command import GreedyStr

try:  # pragma: no cover - 正常 AstrBot 运行时走这里
    import astrbot.api.message_components as Comp
except ImportError:  # pragma: no cover - AstrBot 直接加载 main.py 时的兜底
    from astrbot.api import message_components as Comp  # type: ignore[no-redef]

try:
    from .image_errors import ImageError
    from .image_service import ImageService
    from .image_tools import OpenAIImageTool
except ImportError:  # AstrBot 安装方式可能直接加载 main.py
    from image_errors import ImageError
    from image_service import ImageService
    from image_tools import OpenAIImageTool

# 帮助关键词（大小写不敏感）。
_HELP_TOKENS = {"help", "帮助", "-h", "--help"}
# 图生图关键词；出现在首个 token 时才进入编辑分支。
_EDIT_TOKENS = {"edit", "改图"}
_EDIT_USAGE = "用法：/draw edit <提示词>，并随消息附上一张图片。"
# 非预期异常统一文案：绝不回显异常内容，避免泄露 api_key / 内部细节。
_UNEXPECTED_ERROR = "生图失败，请检查 AstrBot 日志与插件配置。"


class OpenAIImagePlugin(Star):
    """AstrBot OpenAI 生图插件。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.service = ImageService(config)
        if config.get("llm_tool_enabled", True):
            self.context.add_llm_tools(OpenAIImageTool(service=self.service))

    @filter.command("draw", alias={"生图"})
    async def draw(self, event: AstrMessageEvent, args: GreedyStr):
        """OpenAI 生图：/draw <提示词>，或 /draw edit <提示词>（随消息附图），或 /draw help。"""
        text = str(args).strip()
        if not text:
            yield event.plain_result(self.service.format_help())
            return
        if text.lower() in _HELP_TOKENS:
            yield event.plain_result(self.service.format_help())
            return

        parts = text.split(None, 1)
        head = parts[0].lower()
        remainder = parts[1] if len(parts) > 1 else ""
        if head in _EDIT_TOKENS:
            prompt = remainder.strip()
            if not prompt:
                yield event.plain_result(_EDIT_USAGE)
                return
            operation, call = "edit", self.service.edit
        else:
            # 未命中编辑关键词时整段文本都是提示词，保留多词提示词原文。
            prompt, operation, call = text, "generate", self.service.generate

        try:
            self.service.assert_authorized(event)
            paths, _ = await call(event, prompt)
        except Exception as exc:  # noqa: BLE001 - 统一转成用户可读文案
            yield await self._command_error(event, operation, exc)
            return
        yield self._result_chain(event, paths)

    def _result_chain(self, event: AstrMessageEvent, paths: list[str]):
        """只把生成的图片拼成一条消息链返回，不附带任何结果文案。"""
        return event.chain_result([Comp.Image.fromFileSystem(path) for path in paths])

    async def _command_error(
        self, event: AstrMessageEvent, operation: str, exc: Exception
    ):
        """把异常转成用户可读消息；非预期异常只记日志，不回显细节。"""
        if not isinstance(exc, ImageError):
            logger.exception("OpenAI image %s failed", operation)
            message = _UNEXPECTED_ERROR
        else:
            message = str(exc)
        return event.plain_result(message)

    async def terminate(self):
        """插件卸载时释放 HTTP 客户端；失败不影响卸载。"""
        try:
            await self.service.close()
        except Exception as exc:  # noqa: BLE001 - 关闭失败不应中断卸载
            logger.warning("OpenAI image service close failed: %s", type(exc).__name__)
