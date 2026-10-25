"""OpenAI 兼容 Image API 的轻量异步客户端。

设计约束：
- 只依赖 httpx（由 AstrBot 运行时提供）。import 阶段不做网络请求、不读配置。
- api_key 仅保存在实例属性上，绝不写入日志、导出或错误文案。
- 只使用 ``/images/generations`` 与 ``/images/edits`` 两个端点，绝不发送
  response_format / style / user（GPT image 系列会返回 400 unknown_parameter）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any

try:  # httpx 由 AstrBot 运行时提供；缺失时给可读报错而不是 import 崩溃
    import httpx
except ImportError:  # pragma: no cover - 仅在缺依赖的环境触发
    httpx = None  # type: ignore[assignment]

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover - 非 AstrBot 环境（如离线自测）回退到标准库
    logger = logging.getLogger("astrbot")

try:
    from .image_errors import (
        QUOTA_ERROR_CODES,
        ImageAPIError,
        ImageError,
        ImageValidationError,
        describe_api_error,
    )
except ImportError:  # pragma: no cover - 允许以顶层模块方式导入（离线测试）
    from image_errors import (  # type: ignore[no-redef]
        QUOTA_ERROR_CODES,
        ImageAPIError,
        ImageError,
        ImageValidationError,
        describe_api_error,
    )

__all__ = ["ImageCallResult", "ImageResult", "OpenAIImageClient"]

_MAX_RETRY_AFTER = 60.0
_BACKOFF_BASE = 0.5
_BACKOFF_JITTER = 0.3
_NO_RETRY_STATUSES = (400, 401, 403)
# 无论状态码如何都不得重试：内容被安全策略拦截、账户额度/账单已耗尽（不会自愈）。
_NO_RETRY_CODES: frozenset[str] = frozenset({"moderation_blocked"}) | QUOTA_ERROR_CODES

_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_DEFAULT_MIME = "image/png"
_DEFAULT_FILENAME = "image.png"


def _collect_network_errors() -> tuple[type[BaseException], ...]:
    """解析 httpx 中代表网络层失败的异常类型（缺失时退化为 OSError）。"""
    errors: list[type[BaseException]] = [OSError]  # 覆盖内置 TimeoutError
    for name in ("HTTPError", "TimeoutException", "TransportError", "ConnectError"):
        candidate = getattr(httpx, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            errors.append(candidate)
    return tuple(dict.fromkeys(errors))


_NETWORK_ERRORS: tuple[type[BaseException], ...] = _collect_network_errors()


def _status_of(response: object) -> int:
    value = getattr(response, "status_code", None)
    if value is None:
        value = getattr(response, "status", None)
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _get_header(response: object, name: str) -> str:
    headers = getattr(response, "headers", None)
    if not headers:
        return ""
    try:
        value = headers.get(name)
    except (AttributeError, TypeError, ValueError):
        value = None
    if value:
        return str(value).strip()
    try:
        items = list(headers.items())
    except (AttributeError, TypeError, ValueError):
        return ""
    wanted = name.lower()
    for key, value in items:
        if str(key).lower() == wanted and value:
            return str(value).strip()
    return ""


def _response_json(response: object) -> object:
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            return json_method()
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug(f"响应 .json() 解析失败（{type(exc).__name__}），回退到 text")
    text = getattr(response, "text", None)
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8", "replace")
    if isinstance(text, str) and text.strip():
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return None
    return None


def _error_object(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    error = payload.get("error")
    if isinstance(error, dict):
        return error
    if isinstance(error, str) and error.strip():
        return {"message": error.strip()}
    if any(key in payload for key in ("message", "code", "type", "param")):
        return payload
    return {}


def _error_field(payload: object, field: str) -> str:
    value = _error_object(payload).get(field)
    return value.strip() if isinstance(value, str) else ""


def _as_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _parse_retry_after(value: str) -> float | None:
    """把 Retry-After 响应头解析为秒数；非数字（如 HTTP 日期）返回 None。"""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except (TypeError, ValueError):
        return None
    if seconds < 0 or math.isnan(seconds):
        return None
    return seconds


def _guess_mime(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return _MIME_BY_EXT.get(ext, _DEFAULT_MIME)


def _file_part(item: tuple[str, bytes]) -> tuple[str, bytes, str]:
    """把 (filename, data) 转成 httpx files 需要的 (filename, data, mime)。"""
    name, data = item
    filename = (name or "").strip() or _DEFAULT_FILENAME
    return filename, data, _guess_mime(filename)


def _put_optional(target: dict[str, Any], **fields: Any) -> None:
    """只把非空字符串参数放进请求体（空值一律剔除）。"""
    for key, value in fields.items():
        if isinstance(value, str):
            value = value.strip()
        if value:
            target[key] = value


@dataclass
class ImageResult:
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


@dataclass
class ImageCallResult:
    images: list[ImageResult]
    usage: dict
    request_id: str
    status: int


class OpenAIImageClient:
    """异步调用 OpenAI 兼容的 Image API。

    参数:
        base_url: 形如 ``https://api.openai.com/v1``，尾部 ``/`` 会被去掉。
        api_key: 凭据，只保存在实例上，不写日志。
        timeout: 单次请求超时（秒）。
        max_retries: 429 / 5xx / 网络异常的最大重试次数。
        http_client: 可注入的 httpx.AsyncClient（或同接口的假对象）；
            为 None 时首次请求惰性创建并由本类负责关闭。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 180.0,
        max_retries: int = 2,
        http_client: object | None = None,
    ) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = api_key or ""
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self._client = http_client
        self._owns_client = False

    # ------------------------------------------------------------------ 基础

    @property
    def generations_url(self) -> str:
        return f"{self.base_url}/images/generations"

    @property
    def edits_url(self) -> str:
        return f"{self.base_url}/images/edits"

    def _ensure_client(self) -> Any:
        client = self._client
        if client is None:
            if httpx is None:
                raise ImageError("httpx 不可用，无法创建 HTTP 客户端")
            client = httpx.AsyncClient(timeout=self.timeout)
            self._client = client
            self._owns_client = True
        return client

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _scrub(self, text: str) -> str:
        """万一上游把凭据回显在错误里，也要从文案中抹掉。"""
        key = self.api_key
        if key and len(key) >= 6 and key in text:
            return text.replace(key, "***")
        return text

    def _backoff_delay(self, attempt: int) -> float:
        delay = _BACKOFF_BASE * (2**attempt) + random.uniform(0, _BACKOFF_JITTER)
        return min(delay, _MAX_RETRY_AFTER)

    def _retry_delay(
        self, attempt: int, status: int, retry_after: float | None
    ) -> float:
        if status == 429 and retry_after is not None:
            return min(max(retry_after, 0.0), _MAX_RETRY_AFTER)
        return self._backoff_delay(attempt)

    def _should_retry(self, status: int, code: str) -> bool:
        # 额度/账单类错误与内容安全拦截无论状态码如何都不重试。
        if code.strip().lower() in _NO_RETRY_CODES:
            return False
        if status in _NO_RETRY_STATUSES:
            return False
        return status == 429 or status >= 500

    # ------------------------------------------------------------------ 请求

    async def _post(
        self,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    ) -> object:
        attempt = 0
        while True:
            client = self._ensure_client()
            started = time.monotonic()
            try:
                if json_body is not None:
                    # Content-Type: application/json 由 httpx 依据 json= 自动补上。
                    response = await client.post(
                        url, headers=self._auth_headers(), json=json_body
                    )
                else:
                    response = await client.post(
                        url,
                        headers=self._auth_headers(),
                        data=data,
                        files=files,
                    )
            except _NETWORK_ERRORS as exc:
                elapsed = time.monotonic() - started
                reason = type(exc).__name__
                if attempt >= self.max_retries:
                    logger.warning(
                        f"生图请求网络异常 {reason}，已重试 {attempt} 次仍失败，"
                        f"耗时 {elapsed:.2f}s"
                    )
                    raise ImageAPIError(
                        f"生图请求网络异常（{reason}），重试 {attempt} 次后仍失败。"
                    ) from exc
                delay = self._backoff_delay(attempt)
                logger.warning(
                    f"生图请求网络异常 {reason}，{delay:.2f}s 后重试"
                    f"（第 {attempt + 1}/{self.max_retries} 次）"
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue

            status = _status_of(response)
            elapsed = time.monotonic() - started
            request_id = _get_header(response, "x-request-id")

            if status >= 400:
                payload = _response_json(response)
                code = _error_field(payload, "code")
                error_type = _error_field(payload, "type")
                retry_after = _parse_retry_after(_get_header(response, "retry-after"))
                if self._should_retry(status, code) and attempt < self.max_retries:
                    delay = self._retry_delay(attempt, status, retry_after)
                    logger.warning(
                        f"生图接口返回 {status}（code={code or '-'} "
                        f"request_id={request_id or '-'}），{delay:.2f}s 后重试"
                        f"（第 {attempt + 1}/{self.max_retries} 次）"
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                message = self._scrub(
                    describe_api_error(status, payload, request_id=request_id)
                )
                logger.warning(
                    f"生图接口失败 status={status} code={code or '-'} "
                    f"request_id={request_id or '-'} elapsed={elapsed:.2f}s"
                )
                raise ImageAPIError(
                    message,
                    status=status,
                    code=code,
                    error_type=error_type,
                    request_id=request_id,
                    retry_after=retry_after,
                )

            logger.info(
                f"生图接口成功 status={status} request_id={request_id or '-'} "
                f"elapsed={elapsed:.2f}s"
            )
            return response

    def _parse_response(self, response: object) -> ImageCallResult:
        status = _status_of(response)
        request_id = _get_header(response, "x-request-id")
        payload = _response_json(response)
        if not isinstance(payload, dict):
            raise ImageAPIError(
                f"生图接口返回内容无法解析为 JSON（HTTP {status}）。",
                status=status,
                request_id=request_id,
            )

        images: list[ImageResult] = []
        items = payload.get("data")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                images.append(
                    ImageResult(
                        b64_json=_as_text(item.get("b64_json")),
                        url=_as_text(item.get("url")),
                        revised_prompt=_as_text(item.get("revised_prompt")),
                    )
                )
        if not images:
            raise ImageAPIError(
                "服务端未返回图片数据（data 为空），请检查模型名与参数是否被支持。",
                status=status,
                request_id=request_id,
            )

        usage = payload.get("usage")
        return ImageCallResult(
            images=images,
            usage=usage if isinstance(usage, dict) else {},
            request_id=request_id,
            status=status,
        )

    # ------------------------------------------------------------------ 接口

    async def generate(
        self,
        prompt: str,
        *,
        model: str = "",
        n: int = 1,
        size: str = "",
        quality: str = "",
        background: str = "",
        output_format: str = "",
        moderation: str = "",
        safety_identifier: str = "",
    ) -> ImageCallResult:
        """文生图：POST {base_url}/images/generations（JSON body）。"""
        body: dict[str, Any] = {"prompt": prompt, "n": int(n)}
        _put_optional(
            body,
            model=model,
            size=size,
            quality=quality,
            background=background,
            output_format=output_format,
            moderation=moderation,
            safety_identifier=safety_identifier,
        )
        # 绝不发送 response_format / style / user。
        response = await self._post(self.generations_url, json_body=body)
        return self._parse_response(response)

    async def edit(
        self,
        prompt: str,
        images: list[tuple[str, bytes]],
        *,
        mask: tuple[str, bytes] | None = None,
        model: str = "",
        n: int = 1,
        size: str = "",
        quality: str = "",
        background: str = "",
        output_format: str = "",
        moderation: str = "",
        input_fidelity: str = "",
        safety_identifier: str = "",
    ) -> ImageCallResult:
        """图生图：POST {base_url}/images/edits（multipart/form-data）。"""
        if not images:
            raise ImageValidationError("图生图需要至少一张输入图片。")

        data: dict[str, Any] = {"prompt": prompt, "n": int(n)}
        _put_optional(
            data,
            model=model,
            size=size,
            quality=quality,
            background=background,
            output_format=output_format,
            moderation=moderation,
            input_fidelity=input_fidelity,
            safety_identifier=safety_identifier,
        )

        files: list[tuple[str, tuple[str, bytes, str]]] = [
            ("image[]", _file_part(item)) for item in images
        ]
        if mask is not None:
            files.append(("mask", _file_part(mask)))

        response = await self._post(self.edits_url, data=data, files=files)
        return self._parse_response(response)

    async def download(self, url: str) -> bytes:
        """下载图片 URL 的内容。

        不带 Authorization：URL 通常是指向第三方存储的签名地址，
        携带凭据会把 api_key 泄露给非上游服务。
        """
        client = self._ensure_client()
        started = time.monotonic()
        try:
            response = await client.get(url)
        except _NETWORK_ERRORS as exc:
            logger.warning(f"下载图片网络异常 {type(exc).__name__}")
            raise ImageAPIError(
                f"下载图片失败：网络异常（{type(exc).__name__}）。"
            ) from exc

        status = _status_of(response)
        elapsed = time.monotonic() - started
        request_id = _get_header(response, "x-request-id")
        if status and not 200 <= status < 300:
            payload = _response_json(response)
            message = self._scrub(
                describe_api_error(status, payload, request_id=request_id)
            )
            logger.warning(
                f"下载图片失败 status={status} request_id={request_id or '-'} "
                f"elapsed={elapsed:.2f}s"
            )
            raise ImageAPIError(
                f"下载图片失败（{message}）",
                status=status,
                code=_error_field(payload, "code"),
                error_type=_error_field(payload, "type"),
                request_id=request_id,
            )

        content = getattr(response, "content", None)
        if isinstance(content, bytearray):
            content = bytes(content)
        if not isinstance(content, bytes) or not content:
            raise ImageAPIError(
                "下载图片失败：响应内容为空。", status=status, request_id=request_id
            )
        logger.debug(
            f"下载图片成功 bytes={len(content)} status={status} elapsed={elapsed:.2f}s"
        )
        return content

    async def close(self) -> None:
        """关闭自建的 httpx client；注入的 client 不由本类关闭。幂等，不抛异常。"""
        client = self._client
        if client is None or not self._owns_client:
            return
        self._client = None
        self._owns_client = False
        aclose = getattr(client, "aclose", None)
        if not callable(aclose):
            return
        try:
            await aclose()
        except Exception as exc:  # noqa: BLE001 - 关闭失败不应影响调用方
            logger.debug(f"关闭 httpx 客户端时忽略异常：{type(exc).__name__}")
