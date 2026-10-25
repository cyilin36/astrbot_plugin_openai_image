# astrbot_plugin_openai_image

AstrBot 生图插件：调用 OpenAI 兼容的 Image API 生成与编辑图片，支持指令和 LLM 工具调用。

## 功能

- `/draw [参数] <提示词>` 文生图（别名 `/生图`）
- `/draw edit [参数] <提示词>` 图生图编辑，图片随消息一起发出
- `/draw help` 查看用法
- LLM 工具 `openai_image`：`action=generate` / `action=edit`，由大模型自动调用，结果直接发到当前会话

## 安装

把插件目录放到 AstrBot 的 `data/plugins/` 下，目录名为 `astrbot_plugin_openai_image`，然后在 WebUI 插件页重载或启用。依赖 `httpx`（AstrBot 已自带）。

## 配置

在 WebUI 插件配置页填写：

| 配置项 | 说明 |
| --- | --- |
| `base_url` | 接口地址，填到 `/v1` 为止，例如 `https://api.openai.com/v1`，也可填兼容中转地址 |
| `api_key` | API Key，只用于调用生图接口，不会写入消息或日志 |
| `model` | **必填**，你的服务商实际支持的模型名（如 `gpt-image-2.5-flare`、`gpt-image-2`、`gpt-image-1`、`dall-e-3`） |
| `authorized_uids` | 允许使用的 AstrBot UID 列表，为空时拒绝所有生图请求 |
| `allow_admin` | 是否允许管理员绕过 UID 白名单，默认开启 |
| `default_size` | 默认尺寸，默认 `1024x1024`，也可填 `1536x1024`、`1024x1536`、`auto` 或 `宽x高` 自定义 |
| `default_quality` | 默认画质，默认 `auto`：`low` / `medium` / `high` / `auto`，部分模型支持 `xhigh` / `max` |
| `output_format` | 输出格式：`png` / `jpeg` / `webp` |
| `background` | 背景：`auto` / `transparent` / `opaque`（透明背景仅支持 png / webp） |
| `moderation` | 内容审核级别：`auto` / `low` |
| `default_n` / `max_n` | 默认生成张数与单次张数上限 |
| `timeout_seconds` | 单次请求超时，默认 180 秒 |
| `max_retries` | 限流与服务端错误的重试次数 |
| `max_concurrent` | 同时进行的生图请求数上限 |
| `cooldown_seconds` | 同一用户的调用冷却时间，0 表示不限制 |
| `llm_tool_enabled` | 是否向大模型暴露 `openai_image` 工具 |
| `save_dir` | 图片保存目录，留空则存到 `data/plugin_data/astrbot_plugin_openai_image` |

`base_url`、`api_key`、`model` 三项未配置时不会生图，会直接提示缺哪一项。

## 使用

```
/draw 一只戴墨镜的橘猫
/draw --size 1536x1024 --quality high 一只戴墨镜的橘猫
/draw edit 把背景换成星空        # 需要把图片和这条消息一起发出
/draw help
/生图 -s 1024x1536 -q high 星空下的海边小屋   # 别名，用法相同
```

参数写在提示词前面，支持 `--size` / `-s`、`--quality` / `-q`、`--n`、`--background`、`--format`、`--model`，也可以写成 `size=...` 形式：

- `size`：默认 `1024x1024`，可用 `1536x1024`、`1024x1536`、`auto` 或自定义 `宽x高`（两边为 16 的倍数、宽高比 1:3~3:1、最长边不超过 3840）。
- `quality`：默认 `auto`，可用 `low` / `medium` / `high`，部分模型支持 `xhigh` / `max`。

参数只从提示词之前解析，遇到第一个非参数词即停止，所以提示词中间出现的 `xxx=yyy` 仍原样保留。未指定的参数使用上面的插件配置；LLM 工具 `openai_image` 同样支持这些字段。

获取自己的 UID：在聊天中发送 `/sid`，把返回的 UID 填进 `authorized_uids`。

## 说明

- 生成结果保存在插件数据目录，并按日期分子目录。
- 内容被安全策略拦截时不会重试，请修改提示词或更换图片。
- 限流（429）与服务端错误（5xx）会自动退避重试；额度不足不会重试。