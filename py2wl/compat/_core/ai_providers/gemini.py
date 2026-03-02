#!/usr/bin/env python3
import os
import requests
from .base import AIProvider


class GeminiProvider(AIProvider):
    """Google Gemini — py2wl AI 插件实现。"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供 GOOGLE_API_KEY")
        self.model = "gemini-1.5-flash"
        self.api_endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        url = f"{self.api_endpoint}?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.2),
                "maxOutputTokens": kwargs.get("max_tokens", 256),
            },
        }
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

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
