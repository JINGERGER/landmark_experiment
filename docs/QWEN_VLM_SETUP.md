# Qwen VLM (Aliyun 百炼) 集成与使用说明

本项目支持通过 Qwen VLM (阿里云百炼) 作为视觉语言模型（VLM）后端进行地标智能关联。

## 1. 依赖安装

请确保已安装 Aliyun 兼容的 `openai` Python SDK：

```bash
pip install dashscope-openai
```

> 注意：如已安装官方 openai 包，建议新建虚拟环境，避免冲突。

## 2. API Key 获取与配置

- 登录 [阿里云百炼控制台](https://dashscope.console.aliyun.com/) 获取 API Key。
- 北京地域和新加坡地域的 API Key 不同，请根据实际选择。
- 推荐将 API Key 设置为环境变量：

```bash
export DASHSCOPE_API_KEY=sk-xxxxxx
```

## 3. 模型参数设置

- 支持的模型名称如：`qwen3-vl-plus`，可在[官方模型列表](https://help.aliyun.com/zh/model-studio/models)查询。
- 在 ROS2 启动参数或配置文件中设置：

```yaml
vlm_model_name: qwen3-vl-plus
```

## 4. 代码调用示例

集成代码片段如下：

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen3-vl-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "<你的VLM提示内容>"},
    ],
    max_tokens=500,
    temperature=0.1
)
print(completion.choices[0].message.content)
```

> 在本项目中，已自动适配 Qwen VLM，设置模型名为 `qwen` 前缀即可。

## 5. 常见问题

- **ImportError: openai could not be resolved**
  - 请确认已安装 dashscope-openai 包。
- **API Key 无效或权限不足**
  - 检查 API Key 是否正确、是否有对应地域和模型的权限。
- **网络连接失败**
  - 检查本地网络与阿里云 dashscope 域名的连通性。
- **模型响应慢或超时**
  - 可适当调整 `timeout` 参数，或联系阿里云支持。

## 6. 参考链接

- [阿里云百炼文档](https://help.aliyun.com/zh/model-studio/)
- [API Key 获取说明](https://help.aliyun.com/zh/model-studio/get-api-key)
- [模型列表](https://help.aliyun.com/zh/model-studio/models)

---
如有更多问题，请参考官方文档或联系项目维护者。
