#!/usr/bin/env python3
from abc import ABC, abstractmethod


class AIProvider(ABC):
    """
    AI 服务提供商抽象基类。
    py2wl 使用场景：
      - suggest_mapping : 根据 Python 函数路径推断 Wolfram 等价函数名
      - rerank_candidates: 对预筛候选列表按语义相关度重排
      - explain_mapping  : 向用户解释某条 Python→WL 映射的含义与注意事项
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """向模型发送原始 prompt，返回文本响应。"""
        pass

    @abstractmethod
    def explain_mapping(self, python_path: str, rule: dict) -> str:
        """
        解释一条 Python→Wolfram 映射规则。

        Args:
            python_path : 如 "numpy.linalg.eig"
            rule        : YAML 映射规则 dict，含 wolfram_function / description / tags 等
        Returns:
            面向开发者的说明文字（可含注意事项 / 坑）
        """
        pass
