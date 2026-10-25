"""生图插件的异常类型与错误文案映射。

本模块不依赖第三方库，也不做任何 I/O，可在任意环境安全导入。
文案中不会包含 api_key（本模块也拿不到 api_key）。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "QUOTA_ERROR_CODES",
    "ImageAPIError",
    "ImageAuthError",
    "ImageConfigError",
    "ImageError",
    "ImageValidationError",
    "describe_api_error",
]

# 额度 / 账单类错误：账户状态不会因等待而自愈，客户端不得重试。
# 单一来源：image_client._should_retry 复用本集合。
QUOTA_ERROR_CODES = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        "billing_hard_limit_reached",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
    }
)

# 上游 error.message 可能很长，兜底文案里做一次裁剪。
_MAX_MESSAGE_CHARS = 300


class ImageError(Exception):
    """生图插件相关错误的基类。"""


class ImageConfigError(ImageError):
    """配置缺失或不合法（未配置 base_url / api_key / model 等）。"""


class ImageAuthError(ImageError):
    """调用者 UID 未授权使用本插件。"""


class ImageValidationError(ImageError):
    """调用参数非法（prompt 为空或超长、n 超限、mask 非法等）。"""


class ImageAPIError(ImageError):
    """上游 Image API 返回错误，或请求在网络层失败。

    公开属性：
        message: 可直接展示给用户的中文文案。
        status: HTTP 状态码，未知时为 0。
        code: 上游 error.code。
        error_type: 上游 error.type。
        request_id: 上游响应头 x-request-id。
        retry_after: 建议等待秒数（float）或 None。
    """

    def __init__(
        self,
        message: str,
        *,
        status: int = 0,
        code: str = "",
        error_type: str = "",
        request_id: str = "",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = int(status or 0)
        self.code = code or ""
        self.error_type = error_type or ""
        self.request_id = request_id or ""
        self.retry_after = retry_after


def _clip(text: str, limit: int = _MAX_MESSAGE_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _extract_error(payload: object) -> dict[str, Any]:
    """从已解析的响应 JSON 中取出 error 对象。

    兼容三种形态：``{"error": {...}}``、``{"error": "文本"}``、
    以及服务端把 error 字段直接平铺在顶层的情况。
    """
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


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None or isinstance(value, (dict, list)):
        return ""
    return str(value).strip()


def describe_api_error(
    status: int,
    payload: object,
    *,
    request_id: str = "",
) -> str:
    """把上游错误响应翻译成中文可读文案（按 error.code 优先分支）。"""
    error = _extract_error(payload)
    code = _as_text(error.get("code")).lower()
    error_type = _as_text(error.get("type")).lower()
    message = _as_text(error.get("message"))
    param = _as_text(error.get("param"))
    http_status = int(status or 0)

    if code == "moderation_blocked":
        text = (
            "提示词或图片被安全策略拦截（moderation_blocked），"
            "请修改提示词或更换图片后重试。"
        )
    elif code == "unknown_parameter":
        if param:
            text = (
                f"参数 {param} 不被当前模型支持（unknown_parameter），"
                "请移除该参数或更换模型。"
            )
        else:
            text = "请求包含当前模型不支持的参数（unknown_parameter），请移除该参数或更换模型。"
    elif code in ("invalid_api_key", "invalid_organization") or http_status == 401:
        text = (
            f"API Key 无效或未配置（HTTP {http_status or 401}），"
            "请检查插件配置中的 api_key 是否正确、是否过期。"
        )
    elif code in QUOTA_ERROR_CODES:
        # 额度类错误不会自愈，客户端不重试，文案也不引导用户「稍后重试」。
        text = (
            f"账户额度不足或已达账单 / 用量上限（HTTP {http_status or 429}）："
            "请检查账户余额与用量限制（此类错误不会自动重试）。"
        )
    elif code == "rate_limit_exceeded" or http_status == 429:
        text = (
            f"请求过于频繁被限流（HTTP {http_status or 429}）："
            "请稍后重试（已按 Retry-After 自动重试）。"
        )
    elif (
        code
        in (
            "country_not_supported",
            "unsupported_country_region_territory",
            "permission_denied",
        )
        or http_status == 403
    ):
        text = (
            f"当前地区不支持或组织未完成验证（HTTP {http_status or 403}），"
            "请更换网络地区或完成组织验证。"
        )
    elif http_status >= 500 or error_type == "server_error":
        text = f"生图服务端错误（HTTP {http_status}），可稍后重试。"
    else:
        status_hint = f"HTTP {http_status}" if http_status else "HTTP 状态未知"
        if message:
            text = f"生图接口调用失败（{status_hint}）：{_clip(message)}"
        else:
            text = f"生图接口调用失败（{status_hint}），服务端未提供错误详情。"

    if request_id:
        text = f"{text}（请求 ID: {request_id}）"
    return text
