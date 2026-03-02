#!/usr/bin/env python3
"""
rule_editor.py — 规则编辑器
============================
以结构化方式修改 YAML 规则，杜绝正则/字符串拼接。

用法（命令行）：
  python rule_editor.py get  scipy.integrate.quad
  python rule_editor.py set  scipy.integrate.quad wolfram_function "{NIntegrate[#1,{x,#2,#3}],0.0}&"
  python rule_editor.py set  scipy.integrate.quad numeric true
  python rule_editor.py del  scipy.integrate.quad notes
  python rule_editor.py list scipy                     # 列出某库所有规则
  python rule_editor.py find "NIntegrate"              # 按 wolfram_function 搜索
  python rule_editor.py check                          # 验证所有 YAML 合法性

用法（Python API）：
  from rule_editor import RuleEditor
  ed = RuleEditor()
  ed.set("scipy.integrate.quad", wolfram_function="{NIntegrate[#1,{x,#2,#3}],0.0}&")
  ed.set("numpy.mean", numeric=True)
  ed.delete_field("scipy.integrate.quad", "notes")
  ed.save()   # 写回原 YAML 文件
"""

import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

MAPPINGS_DIR = Path(__file__).parent / "py2wl" / "compat" / "mappings"


# ── YAML 往返保持注释的 Dumper ──────────────────────────────
class _Dumper(yaml.Dumper):
    """保持字典键顺序，字符串中有特殊字符时自动加引号。"""
    def represent_str(self, s):
        # 含 WL 特殊字符的字符串强制用引号
        if any(c in s for c in '[]{}&#@&|>'):
            return self.represent_scalar('tag:yaml.org,2002:str', s, style='"')
        return self.represent_scalar('tag:yaml.org,2002:str', s)

_Dumper.add_representer(str, _Dumper.represent_str)


def _load_file(path: Path) -> List[Dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else []


def _save_file(path: Path, rules: List[Dict]) -> None:
    # 保持每条规则之间有空行，可读性好
    lines = []
    for rule in rules:
        dumped = yaml.dump(
            [rule], Dumper=_Dumper,
            allow_unicode=True, default_flow_style=False,
            sort_keys=False
        )
        lines.append(dumped.rstrip())
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def _find_file(python_path: str) -> Optional[Path]:
    """根据 python_path 猜测所在 YAML 文件。"""
    lib = python_path.split(".")[0]
    candidates = [
        MAPPINGS_DIR / f"{lib}.yaml",
        MAPPINGS_DIR / f"{lib}_extra.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    # 全搜
    for f in sorted(MAPPINGS_DIR.glob("*.yaml")):
        rules = _load_file(f)
        if any(r.get("python_path") == python_path for r in rules):
            return f
    return None


# ── 公共 API ────────────────────────────────────────────────

class RuleEditor:
    def __init__(self, mappings_dir: Path = MAPPINGS_DIR):
        self._dir = Path(mappings_dir)
        self._cache: Dict[Path, List[Dict]] = {}

    def _get_rules(self, path: Path) -> List[Dict]:
        if path not in self._cache:
            self._cache[path] = _load_file(path)
        return self._cache[path]

    def _find(self, python_path: str):
        """返回 (file_path, rule_dict) 或 (None, None)。"""
        for f in sorted(self._dir.glob("*.yaml")):
            for rule in self._get_rules(f):
                if rule.get("python_path") == python_path:
                    return f, rule
        return None, None

    # ── 读 ──────────────────────────────────────────────────
    def get(self, python_path: str) -> Optional[Dict]:
        _, rule = self._find(python_path)
        return rule

    def list_lib(self, lib: str) -> List[Dict]:
        results = []
        for f in sorted(self._dir.glob("*.yaml")):
            for rule in self._get_rules(f):
                if rule.get("python_path", "").startswith(lib):
                    results.append(rule)
        return results

    def find_wf(self, keyword: str) -> List[Dict]:
        results = []
        for f in sorted(self._dir.glob("*.yaml")):
            for rule in self._get_rules(f):
                if keyword in str(rule.get("wolfram_function", "")):
                    results.append(rule)
        return results

    # ── 写 ──────────────────────────────────────────────────
    def set(self, python_path: str, **fields) -> bool:
        """
        更新规则的一个或多个字段。字段不存在时自动添加。
        返回 True 表示成功，False 表示规则未找到。

        示例：
          ed.set("scipy.integrate.quad",
                 wolfram_function="{NIntegrate[#1,{x,#2,#3}],0.0}&")
          ed.set("numpy.mean", numeric=True)
        """
        f, rule = self._find(python_path)
        if rule is None:
            print(f"❌ 未找到规则：{python_path}", file=sys.stderr)
            return False
        for k, v in fields.items():
            rule[k] = v
        return True

    def delete_field(self, python_path: str, field: str) -> bool:
        """删除规则中的某个字段。"""
        f, rule = self._find(python_path)
        if rule is None:
            return False
        rule.pop(field, None)
        return True

    def add_rule(self, yaml_file: str, **rule_fields) -> bool:
        """向指定 YAML 文件追加一条新规则。"""
        if "python_path" not in rule_fields:
            raise ValueError("新规则必须包含 python_path")
        path = self._dir / yaml_file
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        existing = self._get_rules(path)
        if any(r.get("python_path") == rule_fields["python_path"] for r in existing):
            print(f"⚠️  规则已存在：{rule_fields['python_path']}", file=sys.stderr)
            return False
        existing.append(dict(rule_fields))
        return True

    def remove_rule(self, python_path: str) -> bool:
        """从 YAML 文件中删除整条规则。"""
        f, _ = self._find(python_path)
        if f is None:
            return False
        rules = self._get_rules(f)
        before = len(rules)
        self._cache[f] = [r for r in rules if r.get("python_path") != python_path]
        return len(self._cache[f]) < before

    # ── 持久化 ──────────────────────────────────────────────
    def save(self, dry_run: bool = False) -> Dict[Path, int]:
        """
        将所有改动写回对应 YAML 文件。
        dry_run=True 时只打印改动，不写文件。
        返回 {文件: 修改条数} 字典。
        """
        stats = {}
        for f, rules in self._cache.items():
            original = _load_file(f)
            if rules != original:
                stats[f] = abs(len(rules) - len(original)) or 1
                if not dry_run:
                    _save_file(f, rules)
                    print(f"✅ 已写入 {f.name}  ({len(rules)} 条规则)")
                else:
                    print(f"[dry-run] 将写入 {f.name}  ({len(rules)} 条规则)")
        if not stats:
            print("（无改动）")
        return stats

    # ── 批量操作 ────────────────────────────────────────────
    def batch_set(self, updates: List[Dict]) -> int:
        """
        批量更新。每个 dict 必须有 "python_path" 键，其余为要更新的字段。

        示例：
          ed.batch_set([
              {"python_path": "scipy.integrate.quad",
               "wolfram_function": "{NIntegrate[#1,{x,#2,#3}],0.0}&"},
              {"python_path": "numpy.mean", "numeric": True},
          ])
        """
        count = 0
        for update in updates:
            pp = update.pop("python_path")
            if self.set(pp, **update):
                count += 1
            update["python_path"] = pp  # 恢复
        return count

    # ── 验证 ────────────────────────────────────────────────
    def check(self) -> List[str]:
        """检查所有规则的必填字段和常见问题，返回问题列表。"""
        issues = []
        REQUIRED = {"python_path", "wolfram_function", "output_converter"}
        VALID_OC  = {"from_wxf", "from_wl_image", "from_wxf_dataframe"}

        for f in sorted(self._dir.glob("*.yaml")):
            rules = self._get_rules(f)
            seen  = set()
            for i, r in enumerate(rules):
                pp = r.get("python_path", f"<row {i}>")
                # 必填
                missing = REQUIRED - set(r.keys())
                if missing:
                    issues.append(f"{f.name}:{pp}: 缺少字段 {missing}")
                # 重复
                if pp in seen:
                    issues.append(f"{f.name}:{pp}: 重复规则")
                seen.add(pp)
                # output_converter 合法性
                oc = r.get("output_converter")
                if oc and oc not in VALID_OC:
                    issues.append(f"{f.name}:{pp}: 未知 output_converter={oc!r}")
                # input_converters / input_converter 互斥
                if "input_converter" in r and "input_converters" in r:
                    issues.append(f"{f.name}:{pp}: 同时存在 input_converter 和 input_converters")
        return issues


# ── CLI ─────────────────────────────────────────────────────

def _cli():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    ed = RuleEditor()
    cmd = args[0]

    if cmd == "get" and len(args) == 2:
        rule = ed.get(args[1])
        if rule:
            print(yaml.dump(rule, allow_unicode=True, sort_keys=False))
        else:
            print(f"未找到：{args[1]}", file=sys.stderr)
            sys.exit(1)

    elif cmd == "set" and len(args) >= 4:
        pp = args[1]
        field = args[2]
        # 类型推断
        raw = " ".join(args[3:])
        if raw.lower() == "true":    val = True
        elif raw.lower() == "false": val = False
        else:
            try:   val = int(raw)
            except:
                try:   val = float(raw)
                except: val = raw
        if ed.set(pp, **{field: val}):
            ed.save()
        else:
            sys.exit(1)

    elif cmd == "del" and len(args) == 3:
        if ed.delete_field(args[1], args[2]):
            ed.save()

    elif cmd == "list" and len(args) == 2:
        for r in ed.list_lib(args[1]):
            print(f"  {r['python_path']:<50} → {r.get('wolfram_function','')[:40]}")

    elif cmd == "find" and len(args) == 2:
        for r in ed.find_wf(args[1]):
            print(f"  {r['python_path']:<50} → {r.get('wolfram_function','')[:40]}")

    elif cmd == "check":
        issues = ed.check()
        if issues:
            for issue in issues:
                print(f"  ⚠️  {issue}")
            sys.exit(1)
        else:
            print(f"✅ 全部规则验证通过")

    else:
        print(f"未知命令：{args}", file=sys.stderr)
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
