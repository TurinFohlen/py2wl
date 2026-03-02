#!/usr/bin/env python3
import os
import requests
from .base import AIProvider


class GroqProvider(AIProvider):
    """
    Groq (Llama 3.3 70B) — py2wl AI 插件实现。
    Groq 的推理延迟极低，适合容错系统的实时交互场景。
    使用 REST API 而非 groq 库，保持零额外依赖。
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供 GROQ_API_KEY")
        self.api_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"

    def generate(self, prompt: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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
        return resp.json()["choices"][0]["message"]["content"].strip()

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
