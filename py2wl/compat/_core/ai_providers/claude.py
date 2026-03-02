#!/usr/bin/env python3
import os
import requests
from .base import AIProvider


class ClaudeProvider(AIProvider):
    """Anthropic Claude — py2wl AI 插件实现。"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供 ANTHROPIC_API_KEY")
        self.api_endpoint = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-5-sonnet-20241022"

    def generate(self, prompt: str, **kwargs) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 256),
        }
        resp = requests.post(self.api_endpoint, headers=headers,
                             json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()

    def explain_mapping(self, python_path: str, rule: dict) -> str:
        wf   = rule.get("wolfram_function", "?")
        desc = rule.get("description", "")
        tags = ", ".join(rule.get("tags") or [])
        prompt = (
            f"你是 Wolfram Language 和 Python 科学计算双栖专家。\n"
            f"请用 2-4 句话解释以下映射关系，重点说明两侧语义差异和使用注意事项：\n\n"
            f"  Python  : {python_path}\n"
            f"  Wolfram : {wf}\n"
            f"  简介    : {desc}\n"
            f"  标签    : {tags}\n\n"
            f"如有数值精度差异、索引约定不同、广播行为差异等坑，请务必提醒。"
        )
        return self.generate(prompt, max_tokens=256, temperature=0.3)
