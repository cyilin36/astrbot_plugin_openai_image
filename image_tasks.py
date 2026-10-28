"""生图后台任务登记表。

LLM 工具把生图请求交给本模块后立刻结束当前回合，真正的生图在独立 asyncio
任务里执行；完成后由工具把图片直接发到会话。任务状态只留在内存里，供 LLM
在用户主动询问时查询（工具 action=status）。

本模块刻意不做任何主动推送、不唤醒 LLM：生图完成不会触发模型说话。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - 取决于运行环境是否装了 astrbot
    from astrbot.api import logger as _astrbot_logger
except Exception:  # noqa: BLE001 - astrbot 缺失时回退标准库日志
    _astrbot_logger = None

logger: Any = (
    _astrbot_logger if _astrbot_logger is not None else logging.getLogger(__name__)
)

STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"

# 每个会话保留的任务条数上限（含运行中的）；超出后优先丢弃最旧的已结束任务。
MAX_TASKS_PER_SESSION = 20
# 状态页里提示词的最大展示长度。
MAX_PROMPT_PREVIEW = 80


def _preview(text: str) -> str:
    """截断过长的提示词，避免状态查询把整段提示词带回给模型。"""

    text = str(text or "").strip()
    if len(text) <= MAX_PROMPT_PREVIEW:
        return text
    return text[:MAX_PROMPT_PREVIEW] + "…"


@dataclass
class ImageTask:
    """一次后台生图任务的状态记录。"""

    task_id: str
    session: str
    action: str
    prompt: str
    model: str = ""
    size: str = ""
    created_at: float = field(default_factory=time.monotonic)
    status: str = STATUS_RUNNING
    paths: list[str] = field(default_factory=list)
    error: str = ""
    sent_to_user: bool = False
    finished_at: float = 0.0

    def succeed(self, paths: list[str]) -> None:
        """标记为成功并记录落盘路径。"""

        self.status = STATUS_SUCCEEDED
        self.paths = [str(path) for path in paths]
        self.error = ""
        self.finished_at = time.monotonic()

    def fail(self, error: str) -> None:
        """标记为失败并记录用户可读的失败原因。"""

        self.status = STATUS_FAILED
        self.error = str(error)
        self.finished_at = time.monotonic()

    def elapsed_seconds(self) -> float:
        """已耗时（秒）；运行中的任务按当前时间计算。"""

        end = self.finished_at or time.monotonic()
        return max(0.0, end - self.created_at)

    def to_payload(self) -> dict[str, Any]:
        """转成给 LLM 看的精简结构。"""

        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "action": self.action,
            "status": self.status,
            "prompt": _preview(self.prompt),
            "model": self.model,
            "size": self.size,
            "elapsed_seconds": round(self.elapsed_seconds(), 1),
        }
        if self.status == STATUS_SUCCEEDED:
            payload["paths"] = list(self.paths)
            payload["sent_to_user"] = self.sent_to_user
        if self.error:
            payload["error"] = self.error
        return payload


class ImageTaskRegistry:
    """按会话保存生图任务，并持有正在运行的 asyncio 任务。"""

    def __init__(self, *, max_tasks_per_session: int = MAX_TASKS_PER_SESSION) -> None:
        self.max_tasks_per_session = max(1, int(max_tasks_per_session))
        self._sessions: dict[str, list[ImageTask]] = {}
        self._jobs: dict[str, asyncio.Future] = {}

    # ------------------------------------------------------------------ 生命周期

    def new_task(
        self,
        session: str,
        action: str,
        prompt: str,
        *,
        model: str = "",
        size: str = "",
    ) -> ImageTask:
        """登记一条新任务（状态为 running），返回任务记录。"""

        task = ImageTask(
            task_id=uuid.uuid4().hex[:12],
            session=str(session),
            action=str(action),
            prompt=str(prompt),
            model=str(model),
            size=str(size),
        )
        bucket = self._sessions.setdefault(task.session, [])
        bucket.append(task)
        self._trim(bucket)
        return task

    def start(self, task: ImageTask, job: Any) -> None:
        """把生图协程挂到事件循环上执行；异常一律由协程自己回写任务状态。"""

        future = asyncio.ensure_future(job)
        self._jobs[task.task_id] = future

        def _discard(_future: asyncio.Future, task_id: str = task.task_id) -> None:
            self._jobs.pop(task_id, None)

        future.add_done_callback(_discard)

    def _trim(self, bucket: list[ImageTask]) -> None:
        """超出上限时丢掉最旧的已结束任务；全在运行中就暂不清理。"""

        while len(bucket) > self.max_tasks_per_session:
            for index, task in enumerate(bucket):
                if task.status != STATUS_RUNNING:
                    del bucket[index]
                    break
            else:
                return

    # ------------------------------------------------------------------ 查询

    def list(self, session: str) -> list[ImageTask]:
        """返回该会话的任务列表，最新的在前。"""

        return list(reversed(self._sessions.get(str(session), [])))

    def get(self, session: str, task_id: str) -> ImageTask | None:
        """按 task_id 取某条任务；不属于该会话时返回 None。"""

        wanted = str(task_id or "").strip()
        if not wanted:
            return None
        for task in self._sessions.get(str(session), []):
            if task.task_id == wanted:
                return task
        return None

    def running(self, session: str) -> list[ImageTask]:
        """返回该会话仍在生成中的任务。"""

        return [
            task for task in self.list(session) if task.status == STATUS_RUNNING
        ]

    # ------------------------------------------------------------------ 收尾

    async def shutdown(self) -> None:
        """插件卸载时取消所有运行中的任务，避免遗留协程继续调用接口。"""

        jobs = [job for job in self._jobs.values() if not job.done()]
        self._jobs.clear()
        for job in jobs:
            job.cancel()
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)
