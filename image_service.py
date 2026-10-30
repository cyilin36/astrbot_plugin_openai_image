"""生图插件的业务服务层。

职责：
- 校验调用方权限（authorized_uids / allow_admin）
- 校验并补全生图参数（build_options）
- 调用 image_client.OpenAIImageClient 完成文生图 / 图生图编辑
- 把返回的图片落盘到本地

本模块只在运行期依赖 astrbot（可选）与 image_client / image_errors。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:  # AstrBot 以包方式加载插件：__import__(path, fromlist=["main"])
    from .image_client import ImageCallResult, OpenAIImageClient
    from .image_errors import (
        ImageAPIError,
        ImageAuthError,
        ImageConfigError,
        ImageValidationError,
    )
except ImportError:  # 直接以顶层模块加载时的兜底
    from image_client import ImageCallResult, OpenAIImageClient
    from image_errors import (
        ImageAPIError,
        ImageAuthError,
        ImageConfigError,
        ImageValidationError,
    )

try:  # pragma: no cover - 取决于运行环境是否装了 astrbot
    from astrbot.api import logger as _astrbot_logger
except Exception:  # noqa: BLE001 - astrbot 缺失时回退标准库日志
    _astrbot_logger = None

logger: Any = (
    _astrbot_logger if _astrbot_logger is not None else logging.getLogger(__name__)
)

PLUGIN_NAME = "astrbot_plugin_openai_image"

MAX_PROMPT_LENGTH = 32000
MAX_SIDE = 3840
MIN_ASPECT = 1 / 3
MAX_ASPECT = 3.0
CUSTOM_SIZE_RE = re.compile(r"^(\d+)\s*[xX*×]\s*(\d+)$")

SIZE_WHITELIST = ("1024x1024", "1536x1024", "1024x1536", "auto")
QUALITY_WHITELIST = ("", "low", "medium", "high", "xhigh", "max", "auto")
BACKGROUND_WHITELIST = ("", "auto", "transparent", "opaque")
OUTPUT_FORMAT_WHITELIST = ("png", "jpeg", "webp")
MODERATION_WHITELIST = ("", "auto", "low")

_FORMAT_EXTENSIONS = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "webp": "webp"}

_DATA_URI_RE = re.compile(r"^data:[^;,]*;base64,", re.IGNORECASE)

HELP_TEXT = """\
生图插件使用说明：
- /draw <提示词>      文生图，例如：/draw 一只戴墨镜的橘猫
- /draw edit <提示词> 图生图编辑：把图片随消息一起发出，或引用一张图片，再附带提示词
- /draw help          查看本说明
- 别名：/生图（用法相同，例如 /生图 edit 把背景换成海边）
- LLM 工具调用：模型可自动调用 openai_image 工具完成文生图 / 图生图，
  action=generate 或 action=edit，编辑时图片取自当前消息（没有则取引用消息），
  无需也无法传图片字节。
  工具是后台任务：提交后本轮直接结束（不会回复「正在生成」），图片生成完成后
  自动发到当前会话；需要查进度时让模型用 action=status 查询。

可用参数（写在提示词前面；指令与 LLM 工具都支持）：
  --size / --尺寸       默认 1024x1024；可选 1536x1024、1024x1536、auto，
                        或自定义 宽x高（如 1280x720，两边为 16 的倍数、
                        宽高比 1:3~3:1、最长边不超过 3840）
  --quality / --画质    默认 auto；可选 low、medium、high、xhigh、max
  --n / --张数          生成张数，默认 default_n，上限 max_n
  --background / --背景  auto、transparent、opaque（transparent 仅支持 png / webp）
  --format / --格式     png（默认）、jpeg、webp
  --model / --模型      临时覆盖插件配置里的模型名

示例：
  /draw --size 1536x1024 --quality high 一只戴墨镜的橘猫
  /draw -s 1024x1536 -q high 星空下的海边小屋
  /draw edit --size auto 把背景换成夜晚

参数也可以写成 size=1536x1024 的形式，但必须放在提示词之前；
提示词中间出现的 xxx=yyy 不会被当作参数。参数缺省时使用插件配置：
default_size、default_quality、default_n、background、output_format。
"""


@dataclass
class ImageOptions:
    prompt: str = ""
    model: str = ""
    size: str = ""
    quality: str = ""
    background: str = ""
    output_format: str = ""
    moderation: str = ""
    n: int = 1


def _as_text(value: Any) -> str:
    """把配置值安全地转成去空白字符串（None -> ""）。"""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_int(value: Any, default: int) -> int:
    """把配置值安全地转成 int，失败用默认值。"""

    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    """把配置值安全地转成 float，失败用默认值。"""

    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _normalize_uid_list(value: Any) -> list[str]:
    """把配置里的 uid 白名单统一成 list[str]。

    支持 list / tuple / 逗号（或中文逗号、空白、分号）分隔的字符串。
    """

    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,，;；\s]+", value)
    elif isinstance(value, (list, tuple, set, frozenset)):
        parts = [str(item) for item in value]
    else:
        parts = [str(value)]
    return [item.strip() for item in parts if item and item.strip()]


class ImageService:
    """生图服务：权限校验 / 参数校验 / 调用 / 落盘。"""

    PLUGIN_NAME = PLUGIN_NAME

    def __init__(
        self,
        config: Mapping,
        *,
        data_dir: str | Path | None = None,
        client: OpenAIImageClient | None = None,
    ) -> None:
        self.config: Mapping = config or {}
        self.base_url = _as_text(self.config.get("base_url"))
        self.api_key = _as_text(self.config.get("api_key"))
        self.model = _as_text(self.config.get("model"))
        self.default_size = _as_text(self.config.get("default_size"))
        self.default_quality = _as_text(self.config.get("default_quality"))
        self.output_format = _as_text(self.config.get("output_format"))
        self.background = _as_text(self.config.get("background"))
        self.moderation = _as_text(self.config.get("moderation"))
        self.default_n = _as_int(self.config.get("default_n"), 1)
        self.max_n = _as_int(self.config.get("max_n"), 4)
        self.cooldown_seconds = _as_float(self.config.get("cooldown_seconds"), 0.0)
        self.authorized_uids = _normalize_uid_list(self.config.get("authorized_uids"))
        self.allow_admin = bool(self.config.get("allow_admin"))

        # 目录解析：显式参数 > save_dir 配置 > astrbot 数据目录 > 系统临时目录。
        # 目录本身延迟到落盘时再创建，创建失败也不影响插件加载。
        resolved_dir = data_dir
        if resolved_dir is None:
            save_dir = _as_text(self.config.get("save_dir"))
            if save_dir:
                resolved_dir = save_dir
            else:
                resolved_dir = self._default_data_dir()
        self.data_dir = Path(resolved_dir)

        # 并发控制与冷却记录
        self._semaphore = asyncio.Semaphore(
            max(1, _as_int(self.config.get("max_concurrent"), 1))
        )
        self._last_call_at: dict[str, float] = {}

        # 记录最近一次请求的参数，供落盘扩展名使用
        self._last_model = ""
        self._last_size = ""
        self._last_output_format = ""

        if client is not None:
            self.client: Any = client
        else:
            self.client = OpenAIImageClient(
                self.base_url,
                self.api_key,
                timeout=_as_float(self.config.get("timeout_seconds"), 180.0),
                max_retries=_as_int(self.config.get("max_retries"), 2),
            )

    # ------------------------------------------------------------------ 内部工具

    @staticmethod
    def _default_data_dir() -> Path:
        try:
            from astrbot.core.star.star_tools import StarTools

            return Path(StarTools.get_data_dir(PLUGIN_NAME))
        except Exception:  # noqa: BLE001 - 任何失败都回退临时目录
            fallback = Path(tempfile.gettempdir()) / PLUGIN_NAME
            logger.warning(
                "无法获取 AstrBot 数据目录，回退到 %s 用于保存生成的图片",
                fallback,
            )
            return fallback

    def _get_cfg(self, key: str, default: Any = "") -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:  # noqa: BLE001 - config 可能是任意 Mapping
            return default
        return default if value is None else value

    def _resolve(self, value: Any, *config_keys: str, default: str = "") -> str:
        """参数优先，其次按顺序尝试配置键，最后用内置默认值。"""

        text = _as_text(value)
        if text:
            return text
        for key in config_keys:
            text = _as_text(self._get_cfg(key))
            if text:
                return text
        return default

    def _validate_size(self, size: str) -> str:
        if size in SIZE_WHITELIST:
            return size
        match = CUSTOM_SIZE_RE.match(size)
        if match:
            width, height = int(match.group(1)), int(match.group(2))
            if width > 0 and height > 0 and width % 16 == 0 and height % 16 == 0:
                aspect = width / height
                if (
                    MIN_ASPECT <= aspect <= MAX_ASPECT
                    and max(width, height) <= MAX_SIDE
                ):
                    return size
        raise ImageValidationError(
            f"尺寸 {size!r} 不受支持。可用 1024x1024 / 1536x1024 / 1024x1536 / auto，"
            "或自定义 宽x高（两边为 16 的倍数、宽高比在 1:3~3:1 之间、最长边不超过 3840）"
        )

    # ------------------------------------------------------------------ 权限

    def assert_authorized(self, event: object) -> str:
        """校验事件发送者是否有权使用生图功能，返回 uid。"""

        uid = str(event.get_sender_id())
        if not self.authorized_uids:
            raise ImageAuthError(
                "生图功能尚未授权任何用户：请把你的 UID 填入插件配置 authorized_uids。"
                "在聊天里发送 /sid 可查看自己的 UID。"
            )
        if uid in self.authorized_uids:
            return uid
        is_admin = False
        try:
            is_admin = bool(getattr(event, "is_admin", lambda: False)())
        except Exception:  # noqa: BLE001 - is_admin 可能不存在或行为异常
            is_admin = False
        if self.allow_admin and is_admin:
            return uid
        raise ImageAuthError(
            f"你（UID {uid}）没有生图权限。请把该 UID 加入插件配置 authorized_uids"
            "（可在聊天里发送 /sid 查看自己的 UID）。"
        )

    def _check_cooldown(self, uid: str) -> None:
        cooldown = self.cooldown_seconds
        if cooldown <= 0:
            return
        now = time.monotonic()
        last = self._last_call_at.get(uid)
        if last is not None:
            remaining = cooldown - (now - last)
            if remaining > 0:
                raise ImageValidationError(
                    f"操作过于频繁，请等待 {int(remaining) + 1} 秒后再试"
                )
        self._last_call_at[uid] = now

    # ------------------------------------------------------------------ 参数校验

    def build_options(
        self,
        prompt: str,
        *,
        model: str = "",
        size: str = "",
        quality: str = "",
        n: int = 0,
        background: str = "",
        output_format: str = "",
    ) -> ImageOptions:
        """校验参数并补全默认值；任何非法输入抛 ImageValidationError / ImageConfigError。"""

        text = _as_text(prompt)
        if not text:
            raise ImageValidationError("提示词不能为空")
        if len(text) > MAX_PROMPT_LENGTH:
            raise ImageValidationError(
                f"提示词过长（{len(text)} 字符），上限为 {MAX_PROMPT_LENGTH} 字符"
            )

        resolved_model = _as_text(model) or self.model
        if not resolved_model:
            raise ImageConfigError(
                "尚未配置生图模型：请到插件配置的 model 项填写服务商支持的模型名，"
                "例如 gpt-image-2.5-flare / gpt-image-2 / gpt-image-1 / dall-e-3（以服务商支持为准）"
            )

        if not self.base_url:
            raise ImageConfigError(
                "缺少配置项 base_url：请在插件配置中填写服务商接口地址"
            )
        if not self.api_key:
            raise ImageConfigError(
                "缺少配置项 api_key：请在插件配置中填写服务商 API Key"
            )

        count = _as_int(n, 0)
        if count <= 0:
            count = self.default_n if self.default_n > 0 else 1
        max_n = self.max_n if self.max_n > 0 else 1
        if count > max_n:
            raise ImageValidationError(
                f"生成张数 {count} 超出上限 {max_n}（可用配置项 max_n 调整）"
            )

        resolved_size = self._validate_size(
            self._resolve(size, "default_size", default="1024x1024")
        )

        resolved_quality = self._resolve(quality, "default_quality", default="auto")
        if resolved_quality not in QUALITY_WHITELIST:
            raise ImageValidationError(
                f"quality {resolved_quality!r} 不受支持，可用：auto / low / medium / high / xhigh / max"
            )

        resolved_background = self._resolve(background, "background", default="")
        if resolved_background not in BACKGROUND_WHITELIST:
            raise ImageValidationError(
                f"background {resolved_background!r} 不受支持，可用：auto / transparent / opaque"
            )

        resolved_format = self._resolve(output_format, "output_format", default="png")
        if resolved_format not in OUTPUT_FORMAT_WHITELIST:
            raise ImageValidationError(
                f"output_format {resolved_format!r} 不受支持，可用：png / jpeg / webp"
            )
        if resolved_background == "transparent" and resolved_format not in (
            "png",
            "webp",
        ):
            raise ImageValidationError("transparent 背景仅支持 png 或 webp 输出格式")

        resolved_moderation = _as_text(self._get_cfg("moderation"))
        if resolved_moderation not in MODERATION_WHITELIST:
            resolved_moderation = ""

        return ImageOptions(
            prompt=text,
            model=resolved_model,
            size=resolved_size,
            quality=resolved_quality,
            background=resolved_background,
            output_format=resolved_format,
            moderation=resolved_moderation,
            n=count,
        )

    # ------------------------------------------------------------------ 生成

    def prepare_generate(
        self,
        event: object,
        prompt: str,
        *,
        model: str = "",
        size: str = "",
        quality: str = "",
        n: int = 0,
        background: str = "",
        output_format: str = "",
    ) -> tuple[str, ImageOptions]:
        """权限、冷却与参数校验，返回 ``(uid, 已补全的选项)``；不做网络请求。"""

        uid = self.assert_authorized(event)
        self._check_cooldown(uid)
        opts = self.build_options(
            prompt,
            model=model,
            size=size,
            quality=quality,
            n=n,
            background=background,
            output_format=output_format,
        )
        return uid, opts

    async def prepare_edit(
        self,
        event: object,
        prompt: str,
        *,
        model: str = "",
        size: str = "",
        quality: str = "",
        n: int = 0,
        background: str = "",
        output_format: str = "",
    ) -> tuple[str, ImageOptions, list[tuple[str, bytes]]]:
        """图生图的准备阶段：校验参数并把待编辑图片读进内存。

        图片必须在提交后台任务之前读完：消息里的图片多来自临时文件，事件处理
        结束后可能被清理。
        """

        uid = self.assert_authorized(event)
        self._check_cooldown(uid)
        images = await self.extract_images(event)
        if not images:
            raise ImageValidationError(
                "编辑图片需要随消息附上一张图片，或引用一张图片后再发送指令"
            )
        opts = self.build_options(
            prompt,
            model=model,
            size=size,
            quality=quality,
            n=n,
            background=background,
            output_format=output_format,
        )
        return uid, opts, images

    async def run_generate(
        self, uid: str, opts: ImageOptions
    ) -> tuple[list[str], ImageCallResult]:
        """按已校验的选项调用文生图接口并落盘；后台任务直接用这个方法。"""

        async with self._semaphore:
            result = await self.client.generate(
                opts.prompt,
                model=opts.model,
                n=opts.n,
                size=opts.size,
                quality=opts.quality,
                background=opts.background,
                output_format=opts.output_format,
                moderation=opts.moderation,
            )
            paths = await self._persist(result, uid, output_format=opts.output_format)
        self._remember(opts)
        return paths, result

    async def run_edit(
        self,
        uid: str,
        opts: ImageOptions,
        images: list[tuple[str, bytes]],
    ) -> tuple[list[str], ImageCallResult]:
        """按已校验的选项调用图生图接口并落盘；后台任务直接用这个方法。"""

        async with self._semaphore:
            result = await self.client.edit(
                opts.prompt,
                images,
                model=opts.model,
                n=opts.n,
                size=opts.size,
                quality=opts.quality,
                background=opts.background,
                output_format=opts.output_format,
                moderation=opts.moderation,
            )
            paths = await self._persist(result, uid, output_format=opts.output_format)
        self._remember(opts)
        return paths, result

    async def generate(
        self,
        event: object,
        prompt: str,
        *,
        model: str = "",
        size: str = "",
        quality: str = "",
        n: int = 0,
        background: str = "",
        output_format: str = "",
    ) -> tuple[list[str], ImageCallResult]:
        uid, opts = self.prepare_generate(
            event,
            prompt,
            model=model,
            size=size,
            quality=quality,
            n=n,
            background=background,
            output_format=output_format,
        )
        return await self.run_generate(uid, opts)

    async def edit(
        self,
        event: object,
        prompt: str,
        *,
        model: str = "",
        size: str = "",
        quality: str = "",
        n: int = 0,
        background: str = "",
        output_format: str = "",
    ) -> tuple[list[str], ImageCallResult]:
        uid, opts, images = await self.prepare_edit(
            event,
            prompt,
            model=model,
            size=size,
            quality=quality,
            n=n,
            background=background,
            output_format=output_format,
        )
        return await self.run_edit(uid, opts, images)

    def _remember(self, opts: ImageOptions) -> None:
        self._last_model = opts.model
        self._last_size = opts.size
        self._last_output_format = opts.output_format

    # ------------------------------------------------------------------ 图片输入

    async def extract_images(self, event: object) -> list[tuple[str, bytes]]:
        """从当前消息或引用消息中提取图片，返回 [(filename, bytes)]；无图返回空列表。

        优先使用随消息直接附带的图片；当前消息没有附图时，回退到被引用消息
        （``Reply.chain``）里的图片，因此「引用一张图 + 说明要改成什么」也能图生图。
        """

        try:
            messages: Sequence[Any] = list(event.get_messages() or [])
        except Exception as exc:  # 事件对象形态不定，不向上抛栈
            raise ImageValidationError(f"读取消息内容失败：{exc}") from exc

        collected = await self._collect_images(messages)
        if collected:
            return collected

        # 没有直接附图时，回退到引用消息链里的图片（AstrBot 会把被引用消息的
        # 完整消息链放进 Reply.chain；引用里再套引用时 chain 不会继续展开）。
        for comp in messages:
            if getattr(type(comp), "__name__", "") != "Reply":
                continue
            chain = getattr(comp, "chain", None)
            if not chain:
                continue
            collected = await self._collect_images(chain)
            if collected:
                return collected
        return []

    async def _collect_images(
        self, components: Sequence[Any]
    ) -> list[tuple[str, bytes]]:
        """从一组消息组件里读取图片；单个组件失败只跳过它，不影响其它图片。"""

        collected: list[tuple[str, bytes]] = []
        for comp in components or []:
            if getattr(type(comp), "__name__", "") != "Image":
                continue
            converter = getattr(comp, "convert_to_file_path", None)
            if converter is None:
                logger.warning("跳过无法解析的图片组件（缺少 convert_to_file_path）")
                continue
            try:
                path = await converter()
            except Exception as exc:  # noqa: BLE001 - 单个组件失败不影响其它图片
                logger.warning("解析消息图片失败，已跳过：%s", exc)
                continue
            if not path:
                logger.warning("消息图片未返回本地路径，已跳过")
                continue
            try:
                data = Path(path).read_bytes()
            except Exception as exc:  # noqa: BLE001 - 读盘失败只跳过该图
                logger.warning("读取图片文件失败，已跳过：%s", exc)
                continue
            if not data:
                logger.warning("图片文件为空，已跳过：%s", path)
                continue
            suffix = Path(path).suffix or ".png"
            collected.append((f"image_{len(collected)}{suffix}", data))
        return collected

    # ------------------------------------------------------------------ 落盘

    def _extension(self, output_format: str) -> str:
        fmt = _as_text(output_format) or self._last_output_format or self.output_format
        return _FORMAT_EXTENSIONS.get(fmt.lower(), "png")

    @staticmethod
    def _decode_b64(raw: str) -> bytes | None:
        payload = _DATA_URI_RE.sub("", raw.strip(), count=1)
        try:
            return base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError):
            return None

    async def _persist(
        self,
        result: ImageCallResult,
        uid: str,
        *,
        output_format: str = "",
    ) -> list[str]:
        """把生成结果写入本地目录，返回已保存的路径列表。"""

        ext = self._extension(output_format)
        stamp = int(time.time() * 1000)
        uid_hash = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:8]
        day_dir = self.data_dir / datetime.now().strftime("%Y%m%d")

        paths: list[str] = []
        for index, image in enumerate(getattr(result, "images", None) or []):
            target = day_dir / f"{stamp}_{uid_hash}_{index}.{ext}"
            data: bytes | None = None
            b64 = getattr(image, "b64_json", None)
            url = getattr(image, "url", None)
            if b64:
                data = self._decode_b64(b64)
                if data is None:
                    logger.warning("第 %s 张图片 base64 解码失败，已跳过", index + 1)
                    continue
            elif url:
                try:
                    data = await self.client.download(url)
                except Exception as exc:  # noqa: BLE001 - 下载失败只跳过该图
                    logger.warning("第 %s 张图片下载失败，已跳过：%s", index + 1, exc)
                    continue
            else:
                logger.warning(
                    "第 %s 张图片既没有 b64_json 也没有 url，已跳过", index + 1
                )
                continue

            if not data:
                logger.warning("第 %s 张图片内容为空，已跳过", index + 1)
                continue
            try:
                day_dir.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            except OSError as exc:
                logger.warning("第 %s 张图片写入失败，已跳过：%s", index + 1, exc)
                continue
            paths.append(str(target))

        if not paths:
            raise ImageAPIError(
                "图片已生成但保存失败，请检查插件 save_dir 配置与目录写入权限"
            )
        return paths

    # ------------------------------------------------------------------ 帮助

    @staticmethod
    def format_help() -> str:
        return HELP_TEXT

    # ------------------------------------------------------------------ 收尾

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception as exc:  # noqa: BLE001 - 关闭失败不影响插件卸载
            logger.warning("关闭生图客户端失败：%s", exc)
