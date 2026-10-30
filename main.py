"""AstrBot OpenAI 生图插件入口。

只暴露一条顶层指令 ``/draw``（别名 ``/生图``），子能力由内部手动路由：

- ``/draw [参数] <提示词>``   文生图
- ``/draw edit [参数] <提示词>``  图生图（图片随同一条消息发出，或引用一张图片）
- ``/draw help``             使用说明

参数（``--size`` / ``--quality`` / ``--n`` / ``--background`` / ``--format`` /
``--model``，以及 ``size=...`` 形式）只从提示词之前解析，遇到第一个非参数
token 即停止，因此提示词里的 ``xxx=yyy`` 仍原样保留。

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

import re

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.star.filter.command import GreedyStr

try:  # pragma: no cover - 正常 AstrBot 运行时走这里
    import astrbot.api.message_components as Comp
except ImportError:  # pragma: no cover - AstrBot 直接加载 main.py 时的兜底
    from astrbot.api import message_components as Comp  # type: ignore[no-redef]

try:
    from .image_errors import ImageError, ImageValidationError
    from .image_service import ImageService
    from .image_tasks import ImageTaskRegistry
    from .image_tools import OpenAIImageTool
except ImportError:  # AstrBot 安装方式可能直接加载 main.py
    from image_errors import ImageError, ImageValidationError
    from image_service import ImageService
    from image_tasks import ImageTaskRegistry
    from image_tools import OpenAIImageTool

# 帮助关键词（大小写不敏感）。
_HELP_TOKENS = {"help", "帮助", "-h", "--help"}
# 图生图关键词；出现在首个 token 时才进入编辑分支。
_EDIT_TOKENS = {"edit", "改图"}
_EDIT_USAGE = (
    "用法：/draw edit [参数] <提示词>，并随消息附上一张图片（也可以引用一张图片）。"
)
# 非预期异常统一文案：绝不回显异常内容，避免泄露 api_key / 内部细节。
_UNEXPECTED_ERROR = "生图失败，请检查 AstrBot 日志与插件配置。"

# 注入到系统提示的一条使用约束：模型调用生图工具的那一轮不要说话。
# 工具本身是后台任务（提交后本轮结束、图片自动发送），模型再多说一句
# 「正在生成」就是纯噪音，而同一轮里模型主动写出的文字无法由工具侧拦截，
# 因此这里从提示词层面先约好。
_TOOL_SYSTEM_NOTE = (
    "\n# 生图工具使用约束\n"
    "调用 `openai_image` 工具生成或编辑图片时，该轮不要输出任何文字："
    "不要说「正在生成」「稍等」「马上好」之类的话，图片会在生成完成后自动发到会话。"
    "不要声称图片已经生成完毕；用户没有主动询问进度时，也不要调用 `action=status` 或汇报进度。"
)

# 指令参数：长名 / 中文别名 -> 服务层字段名。
_OPTION_ALIASES = {
    "size": "size",
    "尺寸": "size",
    "quality": "quality",
    "画质": "quality",
    "质量": "quality",
    "n": "n",
    "count": "n",
    "张数": "n",
    "background": "background",
    "bg": "background",
    "背景": "background",
    "format": "output_format",
    "output_format": "output_format",
    "格式": "output_format",
    "model": "model",
    "模型": "model",
}
# 单字母短参数（-s / -q 这类写法）。
_SHORT_OPTIONS = {
    "s": "size",
    "q": "quality",
    "n": "n",
    "b": "background",
    "f": "output_format",
    "m": "model",
}
_OPTION_HINT = (
    "可用参数：--size / --quality / --n / --background / --format / --model"
    "（也可写成 size=... 的形式，需放在提示词之前）。"
)
_EMPTY_PROMPT = (
    "缺少提示词：请在参数之后写出要生成或编辑的内容（发送 /draw help 查看用法）。"
)


def _lookup_option(name: str, *, short: bool = False) -> str | None:
    """把参数名映射为服务层字段名；未知参数返回 None。"""

    key = name.strip().lower()
    if short:
        return _SHORT_OPTIONS.get(key) or _OPTION_ALIASES.get(key)
    return _OPTION_ALIASES.get(key)


def _parse_options(text: str) -> tuple[dict[str, str], str]:
    """解析提示词前面的参数，返回 ``(参数, 剩余文本)``。

    支持 ``--size 1536x1024``、``-s 1536x1024``、``--size=1536x1024``、
    ``size=1536x1024`` 四种写法。遇到第一个非参数 token 即停止，剩余文本
    （保留内部空格）原样作为提示词；``--`` 之后的文本一律视为提示词。
    """

    tokens = list(re.finditer(r"\S+", text))
    options: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index].group()
        if token == "--":
            index += 1
            break
        sep = ""
        inline = ""
        if token.startswith("--"):
            name, sep, inline = token[2:].partition("=")
            key = _lookup_option(name)
        elif token.startswith("-"):
            name, sep, inline = token[1:].partition("=")
            key = _lookup_option(name, short=True)
        else:
            name, sep, inline = token.partition("=")
            key = _lookup_option(name) if sep else None
        if key is None:
            if token.startswith("-"):
                raise ImageValidationError(f"未知参数 {token}。{_OPTION_HINT}")
            break
        if sep:
            if not inline:
                raise ImageValidationError(f"参数 {token} 缺少取值。{_OPTION_HINT}")
            value = inline
        elif index + 1 < len(tokens):
            value = tokens[index + 1].group()
            index += 1
        else:
            raise ImageValidationError(f"参数 {token} 缺少取值。{_OPTION_HINT}")
        options[key] = value
        index += 1
    prompt = text[tokens[index].start() :] if index < len(tokens) else ""
    return options, prompt.strip()


def _count_from(options: dict[str, str]) -> int:
    """读取 --n 并转成整数；未指定时返回 0（由服务层补默认值）。"""

    raw = options.get("n")
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError as exc:
        raise ImageValidationError(f"张数 {raw!r} 不是整数。") from exc


class OpenAIImagePlugin(Star):
    """AstrBot OpenAI 生图插件。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.service = ImageService(config)
        # LLM 工具把生图放到插件自己的后台任务里执行，任务记录同时供
        # action=status 查询；star_context 用于在后台直接把图片发到会话。
        self.tasks = ImageTaskRegistry()
        if config.get("llm_tool_enabled", True):
            self.context.add_llm_tools(
                OpenAIImageTool(
                    service=self.service,
                    registry=self.tasks,
                    star_context=self.context,
                )
            )

    @filter.on_llm_request()
    async def inject_tool_note(self, event: AstrMessageEvent, req: object):
        """向系统提示补一条生图工具使用约束：调用工具的那一轮不要说话。

        工具是后台任务，提交后本轮直接结束、图片自动发送，模型无需（也不该）
        再输出「正在生成」之类的提示。取不到 system_prompt 时静默跳过。
        """

        if not self.config.get("llm_tool_enabled", True):
            return
        try:
            req.system_prompt += _TOOL_SYSTEM_NOTE
        except Exception as exc:  # noqa: BLE001 - 提示注入失败不应中断请求
            logger.warning("OpenAI image 工具提示注入失败：%s", type(exc).__name__)

    @filter.command("draw", alias={"生图"})
    async def draw(self, event: AstrMessageEvent, args: GreedyStr):
        """OpenAI 生图：/draw [参数] <提示词>，或 /draw edit [参数] <提示词>（随消息附图或引用图片），或 /draw help。"""
        text = str(args).strip()
        if not text or text.lower() in _HELP_TOKENS:
            yield event.plain_result(self.service.format_help())
            return
        try:
            options, rest = _parse_options(text)
        except ImageValidationError as exc:
            yield event.plain_result(str(exc))
            return

        rest = rest.strip()
        if rest.lower() in _HELP_TOKENS:
            yield event.plain_result(self.service.format_help())
            return
        if not rest:
            yield event.plain_result(_EMPTY_PROMPT)
            return

        parts = rest.split(None, 1)
        head = parts[0].lower()
        if head in _EDIT_TOKENS:
            tail = parts[1] if len(parts) > 1 else ""
            try:
                edit_options, prompt = _parse_options(tail)
            except ImageValidationError as exc:
                yield event.plain_result(str(exc))
                return
            if not prompt:
                yield event.plain_result(_EDIT_USAGE)
                return
            options = {**options, **edit_options}
            operation, call = "edit", self.service.edit
        else:
            # 未命中编辑关键词时整段文本都是提示词，保留多词提示词原文。
            prompt, operation, call = rest, "generate", self.service.generate

        try:
            count = _count_from(options)
        except ImageValidationError as exc:
            yield event.plain_result(str(exc))
            return

        # 未在指令里指定参数时传空串/0，由服务层按插件配置补默认值；
        # 画质等参数最终兜底为 auto。
        params = {
            "model": options.get("model", ""),
            "size": options.get("size", ""),
            "quality": options.get("quality", ""),
            "n": count,
            "background": options.get("background", ""),
            "output_format": options.get("output_format", ""),
        }

        try:
            self.service.assert_authorized(event)
            paths, _ = await call(event, prompt, **params)
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
        """插件卸载时取消后台生图任务并释放 HTTP 客户端；失败不影响卸载。"""
        try:
            await self.tasks.shutdown()
        except Exception as exc:  # noqa: BLE001 - 取消失败不应中断卸载
            logger.warning("OpenAI image tasks shutdown failed: %s", type(exc).__name__)
        try:
            await self.service.close()
        except Exception as exc:  # noqa: BLE001 - 关闭失败不应中断卸载
            logger.warning("OpenAI image service close failed: %s", type(exc).__name__)
