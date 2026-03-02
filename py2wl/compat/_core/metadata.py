"""
compat/_core/metadata.py
------------------------
元数据仓库：存储 Python 函数路径 → Wolfram 函数的映射规则。

改造自 dict-tree 的 StorageTree，核心改动：
  - 字符级 Trie → 词级路径 Trie（按 "." 分割）
    numpy.fft.fft 存为 ["numpy", "fft", "fft"] 三层
  - 数据源从脚本扫描改为 YAML 映射文件加载
  - 精确查找 + 标签倒排索引 + 描述关键词索引
"""

import yaml
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

log = logging.getLogger("py2wl.compat")


class PathTrieNode:
    """词级路径 Trie 节点（每个节点对应路径的一个段）"""
    def __init__(self):
        self.children: Dict[str, "PathTrieNode"] = {}
        self.is_end: bool = False
        self.rule: Optional[Dict] = None   # 叶节点存储完整映射规则


class MetadataRepository:
    """
    映射规则仓库，单例设计（启动时加载一次 YAML，后续纯内存查找）。

    映射规则 YAML 格式：
        - python_path: numpy.fft.fft
          wolfram_function: Fourier
          input_converter: to_wl_list
          output_converter: from_wl_passthrough
          tags: [fft, fourier, signal]
          description: "Fast Fourier Transform"
    """

    _instance: Optional["MetadataRepository"] = None

    @classmethod
    def get_instance(cls, mappings_dir: str = None) -> "MetadataRepository":
        if cls._instance is None:
            cls._instance = cls(mappings_dir)
        return cls._instance

    def __init__(self, mappings_dir: str = None):
        self._root = PathTrieNode()
        self._tag_index: Dict[str, List[str]]     = defaultdict(list)
        self._keyword_index: Dict[str, List[str]] = defaultdict(list)
        self._all_rules: List[Dict]               = []

        if not mappings_dir:
            # 自动定位：与本文件同级的 ../mappings 目录
            mappings_dir = str(Path(__file__).parent.parent / "mappings")
        self.load_directory(mappings_dir)

    # ── 加载 ──────────────────────────────────────────────────────
    def load_directory(self, mappings_dir: str):
        """加载目录下所有 .yaml / .yml 映射文件"""
        p = Path(mappings_dir)
        if not p.exists():
            log.warning(f"映射目录不存在：{mappings_dir}")
            return
        count = 0
        for f in sorted(p.rglob("*.yaml")) + sorted(p.rglob("*.yml")):
            count += self._load_file(f)
        log.info(f"加载映射规则 {count} 条，来自 {mappings_dir}")

    def _load_file(self, filepath: Path) -> int:
        try:
            with open(filepath, encoding="utf-8") as f:
                rules = yaml.safe_load(f)
            if not isinstance(rules, list):
                log.warning(f"格式错误（应为列表）：{filepath}")
                return 0
            n = 0
            for rule in rules:
                if self._validate_rule(rule):
                    self._insert_rule(rule)
                    n += 1
            return n
        except Exception as e:
            log.error(f"加载 {filepath} 失败：{e}")
            return 0

    @staticmethod
    def _validate_rule(rule: Dict) -> bool:
        return (isinstance(rule, dict)
                and "python_path" in rule
                and "wolfram_function" in rule)

    def _insert_rule(self, rule: Dict):
        """插入一条规则到词级 Trie"""
        path_parts = rule["python_path"].split(".")
        node = self._root
        for part in path_parts:
            if part not in node.children:
                node.children[part] = PathTrieNode()
            node = node.children[part]
        node.is_end = True
        node.rule = rule
        self._all_rules.append(rule)

        # 标签倒排索引
        for tag in rule.get("tags", []):
            self._tag_index[tag.lower()].append(rule["python_path"])

        # 描述关键词倒排索引
        for word in rule.get("description", "").lower().split():
            self._keyword_index[word].append(rule["python_path"])

    # ── 查找 ──────────────────────────────────────────────────────
    def get_rule(self, python_path: str) -> Optional[Dict]:
        """精确查找，O(depth)"""
        node = self._root
        for part in python_path.split("."):
            if part not in node.children:
                return None
            node = node.children[part]
        return node.rule if node.is_end else None

    def search_rules(self, query: str) -> List[Dict]:
        """
        模糊查找：标签匹配 + 关键词匹配 + 前缀匹配，去重后返回。
        """
        seen = set()
        results = []

        def _add(path):
            if path not in seen:
                rule = self.get_rule(path)
                if rule:
                    seen.add(path)
                    results.append(rule)

        q = query.lower()

        # 1. 标签匹配
        for path in self._tag_index.get(q, []):
            _add(path)

        # 2. 关键词匹配
        for path in self._keyword_index.get(q, []):
            _add(path)

        # 3. python_path 前缀匹配（词级）
        parts = q.split(".")
        node = self._root
        for part in parts:
            if part not in node.children:
                node = None
                break
            node = node.children[part]
        if node:
            self._collect_rules(node, results, seen)

        return results

    def _collect_rules(self, node: PathTrieNode,
                       results: list, seen: set):
        """递归收集子树中所有规则"""
        if node.is_end and node.rule:
            path = node.rule["python_path"]
            if path not in seen:
                seen.add(path)
                results.append(node.rule)
        for child in node.children.values():
            self._collect_rules(child, results, seen)

    @property
    def all_rules(self) -> List[Dict]:
        return list(self._all_rules)
