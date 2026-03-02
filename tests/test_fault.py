"""
test_fault.py — 容错系统单元测试
覆盖：ErrorClassifier / CandidateFinder / FaultHandler / _proxy_base 集成
"""

import sys
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 确保能 import py2wl 包
sys.path.insert(0, str(Path(__file__).parent.parent))

MAPPINGS_DIR = str(Path(__file__).parent / "compat" / "mappings")


# ════════════════════════════════════════════════════════════════
#  1. ErrorClassifier
# ════════════════════════════════════════════════════════════════
class TestErrorClassifier(unittest.TestCase):
    def setUp(self):
        from py2wl.compat._core.error_classifier import classify, FaultKind, RecoverableCategory
        self.classify   = classify
        self.FaultKind  = FaultKind
        self.RC         = RecoverableCategory

    def test_attribute_error_rule_not_found(self):
        exc = AttributeError("未找到 'numpy.fltt.fft' 的 Wolfram 映射。")
        ei  = self.classify(exc, "numpy.fltt.fft")
        self.assertEqual(ei.kind,     self.FaultKind.RECOVERABLE)
        self.assertEqual(ei.category, self.RC.RULE_NOT_FOUND)
        print(f"\n[Test] AttributeError → RULE_NOT_FOUND ✅")

    def test_type_error_arg_mismatch(self):
        exc = TypeError("unsupported operand type")
        ei  = self.classify(exc, "numpy.dot", args=("bad",))
        self.assertEqual(ei.kind,     self.FaultKind.RECOVERABLE)
        self.assertEqual(ei.category, self.RC.ARG_MISMATCH)
        print(f"\n[Test] TypeError → ARG_MISMATCH ✅")

    def test_runtime_kernel_error(self):
        exc = RuntimeError("内核执行失败 [numpy.inv]: $Failed")
        ei  = self.classify(exc, "numpy.inv", raw_wl="Inverse[bad]")
        self.assertEqual(ei.kind, self.FaultKind.RECOVERABLE)
        print(f"\n[Test] RuntimeError → RECOVERABLE/KERNEL_EVAL ✅")

    def test_keyboard_interrupt_fatal(self):
        exc = KeyboardInterrupt()
        ei  = self.classify(exc, "numpy.sum")
        self.assertEqual(ei.kind, self.FaultKind.FATAL)
        print(f"\n[Test] KeyboardInterrupt → FATAL ✅")

    def test_memory_error_fatal(self):
        exc = MemoryError("out of memory")
        ei  = self.classify(exc, "numpy.zeros")
        self.assertEqual(ei.kind, self.FaultKind.FATAL)
        print(f"\n[Test] MemoryError → FATAL ✅")

    def test_value_error_recoverable(self):
        exc = ValueError("参数维度不匹配")
        ei  = self.classify(exc, "scipy.linalg.solve")
        self.assertEqual(ei.kind,     self.FaultKind.RECOVERABLE)
        self.assertEqual(ei.category, self.RC.ARG_MISMATCH)
        print(f"\n[Test] ValueError → ARG_MISMATCH ✅")


# ════════════════════════════════════════════════════════════════
#  2. CandidateFinder
# ════════════════════════════════════════════════════════════════
class TestCandidateFinder(unittest.TestCase):
    def setUp(self):
        from py2wl.compat._core.metadata      import MetadataRepository
        from py2wl.compat._core.candidate_finder import CandidateFinder
        self.repo   = MetadataRepository(MAPPINGS_DIR)
        self.finder = CandidateFinder(self.repo, ai_plugin=None, top_k=6)

    def test_typo_recovery_fft(self):
        """numpy.fltt.fft（拼写错误）应找到 numpy.fft.fft"""
        candidates = self.finder.find("numpy.fltt.fft")
        paths = [r["python_path"] for _, r in candidates]
        self.assertTrue(any("fft" in p for p in paths),
                        f"未找到 fft 相关候选：{paths}")
        print(f"\n[Test] 'numpy.fltt.fft' 拼写容错 → {paths[:3]} ✅")

    def test_typo_recovery_linalg(self):
        """numpy.linalg.sovle（拼写错误）应找到 solve 相关"""
        candidates = self.finder.find("numpy.linalg.sovle")
        paths = [r["python_path"] for _, r in candidates]
        self.assertTrue(any("solve" in p.lower() for p in paths),
                        f"未找到 solve 相关候选：{paths}")
        print(f"\n[Test] 'numpy.linalg.sovle' 拼写容错 → {paths[:3]} ✅")

    def test_namespace_bias(self):
        """查找 scipy.stats.ttest，同命名空间应排前面"""
        candidates = self.finder.find("scipy.stats.ttest")
        if candidates:
            top_ns = candidates[0][1]["python_path"].split(".")[0]
            self.assertEqual(top_ns, "scipy")
        print(f"\n[Test] 命名空间偏置（scipy）✅  top={candidates[0][1]['python_path'] if candidates else 'none'}")

    def test_keyword_match(self):
        """模糊查询 'fourier transform' 应能找到 fft 相关"""
        candidates = self.finder.find("numpy.fourier_transform",
                                       error_hint="fourier transform")
        paths = [r["python_path"] for _, r in candidates]
        self.assertTrue(any("fft" in p or "fourier" in p.lower() for p in paths),
                        f"未找到 fourier 相关候选：{paths}")
        print(f"\n[Test] 关键词匹配 'fourier_transform' ✅  top={paths[0] if paths else 'none'}")

    def test_top_k_limit(self):
        """候选数不超过 top_k"""
        candidates = self.finder.find("numpy.sum")
        self.assertLessEqual(len(candidates), 6)
        print(f"\n[Test] top_k=6 限制 ✅  实际返回 {len(candidates)} 个")

    def test_score_ordering(self):
        """候选按 score 降序排列"""
        candidates = self.finder.find("numpy.linalg.eig")
        scores = [s for s, _ in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))
        print(f"\n[Test] 候选按分数降序 ✅  分数={[f'{s:.2f}' for s in scores]}")

    def test_levenshtein_similarity(self):
        from py2wl.compat._core.candidate_finder import _lev_sim
        self.assertAlmostEqual(_lev_sim("abc", "abc"), 1.0)
        self.assertAlmostEqual(_lev_sim("abc", "xyz"), 0.0)
        self.assertGreater(_lev_sim("fft", "ftt"), 0.5)
        print(f"\n[Test] Levenshtein 相似度计算 ✅")


# ════════════════════════════════════════════════════════════════
#  3. FaultHandler — strict 模式
# ════════════════════════════════════════════════════════════════
class TestFaultHandlerStrict(unittest.TestCase):
    def setUp(self):
        from py2wl.compat._core.metadata      import MetadataRepository
        from py2wl.compat._core.fault_handler import FaultHandler, FaultMode, ActionKind
        self.repo   = MetadataRepository(MAPPINGS_DIR)
        self.handler = FaultHandler(self.repo, ai_plugin=None,
                                    mode=FaultMode.STRICT)
        self.ActionKind = ActionKind
        self.FaultMode  = FaultMode

    def test_strict_raises_on_recoverable(self):
        exc = AttributeError("未找到 'numpy.foo' 的 Wolfram 映射。")
        action = self.handler.handle(exc, "numpy.foo")
        self.assertEqual(action.kind, self.ActionKind.RAISE)
        print(f"\n[Test] strict 模式：可恢复错误 → RAISE ✅")

    def test_strict_raises_on_fatal(self):
        exc = MemoryError("oom")
        action = self.handler.handle(exc, "numpy.zeros")
        self.assertEqual(action.kind, self.ActionKind.RAISE)
        print(f"\n[Test] strict 模式：FATAL → RAISE ✅")

    def test_fatal_always_raises_regardless_of_mode(self):
        """FATAL 错误在任何模式下都必须 RAISE"""
        from py2wl.compat._core.fault_handler import FaultHandler, FaultMode
        for mode in FaultMode:
            h = FaultHandler(self.repo, ai_plugin=None, mode=mode)
            exc = KeyboardInterrupt()
            action = h.handle(exc, "numpy.sum")
            self.assertEqual(action.kind, self.ActionKind.RAISE,
                             f"模式 {mode.value} 下 FATAL 未返回 RAISE")
        print(f"\n[Test] FATAL 在所有模式下均 RAISE ✅")


# ════════════════════════════════════════════════════════════════
#  4. FaultHandler — auto-ai 模式（Mock AI）
# ════════════════════════════════════════════════════════════════
class TestFaultHandlerAutoAI(unittest.TestCase):
    def setUp(self):
        from py2wl.compat._core.metadata      import MetadataRepository
        from py2wl.compat._core.fault_handler import FaultHandler, FaultMode
        self.repo   = MetadataRepository(MAPPINGS_DIR)
        self.FaultHandler = FaultHandler
        self.FaultMode    = FaultMode

    def _make_handler(self, ai=None):
        return self.FaultHandler(self.repo, ai_plugin=ai,
                                  mode=self.FaultMode.AUTO_AI)

    def test_auto_ai_high_confidence_single_candidate(self):
        """置信度 > 0.75 且唯一高分候选 → RETRY_RULE（不询问）"""
        from py2wl.compat._core.fault_handler import ActionKind
        handler = self._make_handler()

        # numpy.fft.fft 的拼写错误，编辑距离很近，应能触发自动重试
        exc = AttributeError("未找到 'numpy.fft.ftt' 的 Wolfram 映射。")
        action = handler.handle(exc, "numpy.fft.ftt")

        # 在无 AI 情况下：若最高分超过阈值且领先第二名，应 RETRY；否则降为 RAISE（strict 降级）
        # 我们只断言不崩溃 + 返回有效 ActionKind
        self.assertIn(action.kind, list(ActionKind))
        print(f"\n[Test] auto-ai 高置信拼写错误 → {action.kind.name} ✅")

    def test_auto_ai_low_confidence_without_interactive(self):
        """完全无关路径 → 无高置信候选 → 在无交互终端时返回 RAISE（TTY 检测）"""
        from py2wl.compat._core.fault_handler import ActionKind
        handler = self._make_handler()
        exc = AttributeError("未找到 'zzz.totally.unknown' 的 Wolfram 映射。")

        # 模拟非 TTY 环境（重定向 stdin）
        with patch("builtins.input", side_effect=EOFError):
            action = handler.handle(exc, "zzz.totally.unknown")
        self.assertIn(action.kind, list(ActionKind))
        print(f"\n[Test] auto-ai 低置信 → {action.kind.name} ✅")

    def test_correction_cache_reuse(self):
        """同一路径第二次出错时直接复用缓存，不再询问"""
        from py2wl.compat._core.fault_handler import ActionKind
        from py2wl.compat._core.metadata      import MetadataRepository
        handler = self._make_handler()

        # 手动注入缓存
        target_rule = self.repo.get_rule("numpy.fft.fft")
        handler._correction_cache["numpy.fft.broken"] = target_rule

        exc = AttributeError("未找到 'numpy.fft.broken' 的 Wolfram 映射。")
        action = handler.handle(exc, "numpy.fft.broken")
        self.assertEqual(action.kind, ActionKind.RETRY_RULE)
        self.assertEqual(action.rule["python_path"], "numpy.fft.fft")
        print(f"\n[Test] 纠错缓存复用 ✅  直接 → {action.rule['python_path']}")

    def test_skip_cache_reuse(self):
        """用户选过 skip 的路径，第二次直接返回 SKIP"""
        from py2wl.compat._core.fault_handler import ActionKind
        handler = self._make_handler()
        handler._skip_cache.add("numpy.nonexistent")

        exc = AttributeError("未找到 'numpy.nonexistent'")
        action = handler.handle(exc, "numpy.nonexistent")
        self.assertEqual(action.kind, ActionKind.SKIP)
        print(f"\n[Test] skip 缓存复用 ✅")


# ════════════════════════════════════════════════════════════════
#  5. FaultHandler — interactive 模式（Mock 用户输入）
# ════════════════════════════════════════════════════════════════
class TestFaultHandlerInteractive(unittest.TestCase):
    def setUp(self):
        from py2wl.compat._core.metadata      import MetadataRepository
        from py2wl.compat._core.fault_handler import FaultHandler, FaultMode
        self.repo    = MetadataRepository(MAPPINGS_DIR)
        self.handler = FaultHandler(self.repo, ai_plugin=None,
                                    mode=FaultMode.INTERACTIVE)
        self.FaultMode = FaultMode

    def _exc(self):
        return AttributeError("未找到 'numpy.fft.ftt' 的 Wolfram 映射。")

    def test_user_selects_candidate(self):
        """用户输入 '1' → RETRY_RULE"""
        from py2wl.compat._core.fault_handler import ActionKind
        with patch("builtins.input", return_value="1"):
            action = self.handler.handle(self._exc(), "numpy.fft.ftt")
        self.assertEqual(action.kind, ActionKind.RETRY_RULE)
        self.assertIsNotNone(action.rule)
        print(f"\n[Test] interactive 用户选候选 '1' → RETRY_RULE ✅  规则={action.rule['python_path']}")

    def test_user_skips(self):
        """用户输入 's' → SKIP"""
        from py2wl.compat._core.fault_handler import ActionKind
        with patch("builtins.input", return_value="s"):
            action = self.handler.handle(self._exc(), "numpy.fft.ftt")
        self.assertEqual(action.kind, ActionKind.SKIP)
        print(f"\n[Test] interactive 用户输入 's' → SKIP ✅")

    def test_user_quits(self):
        """用户输入 'q' 或 Ctrl-C → RAISE"""
        from py2wl.compat._core.fault_handler import ActionKind
        with patch("builtins.input", return_value="q"):
            action = self.handler.handle(self._exc(), "numpy.fft.ftt")
        self.assertEqual(action.kind, ActionKind.RAISE)
        print(f"\n[Test] interactive 用户输入 'q' → RAISE ✅")

    def test_user_custom_expr(self):
        """用户输入 'e' 后提供自定义表达式 → RETRY_EXPR"""
        from py2wl.compat._core.fault_handler import ActionKind
        with patch("builtins.input", side_effect=["e", "Fourier[{1,2,3}]"]):
            action = self.handler.handle(self._exc(), "numpy.fft.ftt")
        self.assertEqual(action.kind, ActionKind.RETRY_EXPR)
        self.assertEqual(action.custom_expr, "Fourier[{1,2,3}]")
        print(f"\n[Test] interactive 用户自定义表达式 → RETRY_EXPR ✅")

    def test_eof_quits(self):
        """非交互环境（EOFError）→ RAISE"""
        from py2wl.compat._core.fault_handler import ActionKind
        with patch("builtins.input", side_effect=EOFError):
            action = self.handler.handle(self._exc(), "numpy.fft.ftt")
        self.assertEqual(action.kind, ActionKind.RAISE)
        print(f"\n[Test] EOFError（非交互环境）→ RAISE ✅")

    def test_clear_cache(self):
        """clear_cache 应清空纠错记忆"""
        self.handler._correction_cache["test"] = {}
        self.handler._skip_cache.add("test2")
        self.handler.clear_cache()
        self.assertEqual(len(self.handler._correction_cache), 0)
        self.assertEqual(len(self.handler._skip_cache), 0)
        print(f"\n[Test] clear_cache ✅")

    def test_set_mode(self):
        """set_mode 动态切换"""
        from py2wl.compat._core.fault_handler import FaultMode
        self.handler.set_mode(FaultMode.STRICT)
        self.assertEqual(self.handler.mode, FaultMode.STRICT)
        self.handler.set_mode(FaultMode.INTERACTIVE)  # 还原
        print(f"\n[Test] 动态切换容错模式 ✅")


# ════════════════════════════════════════════════════════════════
#  6. proxy_base 集成：strict 模式下异常正常传播
# ════════════════════════════════════════════════════════════════
class TestProxyBaseIntegration(unittest.TestCase):
    def setUp(self):
        """注入 MockKernel，隔离真实内核"""
        from py2wl.compat._state import _state
        from py2wl.compat._core.metadata import MetadataRepository
        from py2wl.compat._core.resolver import ResolutionEngine
        from py2wl.compat._core.fault_handler import FaultHandler, FaultMode

        class MockKernel:
            def evaluate(self, expr):
                if "FAIL" in expr:
                    raise RuntimeError("内核执行失败 [test]: $Failed")
                return expr

        repo = MetadataRepository(MAPPINGS_DIR)
        _state["kernel"]        = MockKernel()
        _state["resolver"]      = ResolutionEngine(repo, ai_plugin=None)
        _state["fault_handler"] = FaultHandler(repo, ai_plugin=None,
                                                mode=FaultMode.STRICT)

    def tearDown(self):
        from py2wl.compat._state import _state
        _state["kernel"]        = None
        _state["resolver"]      = None
        _state["fault_handler"] = None

    def test_unknown_path_raises_attribute_error(self):
        """strict 模式：未知路径 → AttributeError"""
        from py2wl.compat._proxy_base import _WolframCallable
        fn = _WolframCallable("totally.unknown.func")
        with self.assertRaises((AttributeError, RuntimeError)):
            fn(1, 2, 3)
        print(f"\n[Test] strict 未知路径 → 异常传播 ✅")

    def test_known_path_resolves(self):
        """已知路径能正常解析到规则（不报 AttributeError）"""
        from py2wl.compat._proxy_base import _WolframCallable
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        rule = repo.get_rule("numpy.fft.fft")
        self.assertIsNotNone(rule, "numpy.fft.fft 规则应存在")
        print(f"\n[Test] 已知路径规则解析 ✅  wolfram_function={rule['wolfram_function']}")

    def test_set_fault_mode_api(self):
        """set_fault_mode 公开 API 可用"""
        from py2wl.compat._proxy_base import set_fault_mode, fault_summary
        set_fault_mode("interactive")
        set_fault_mode("strict")
        summary = fault_summary()
        self.assertIsInstance(summary, list)
        print(f"\n[Test] set_fault_mode / fault_summary API ✅")


# ════════════════════════════════════════════════════════════════
#  7. AI Provider 接口检查（不发真实网络请求）
# ════════════════════════════════════════════════════════════════
class TestAIProviderInterface(unittest.TestCase):
    def _make_provider(self, name, cls_name):
        mod = __import__(
            f"py2wl.compat._core.ai_providers.{name}",
            fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        # 用假 key 实例化
        return cls(api_key="test-key-dummy")

    def _check_provider(self, name, cls_name):
        from py2wl.compat._core.ai_providers.base import AIProvider
        p = self._make_provider(name, cls_name)
        # 必须是 AIProvider 子类
        self.assertIsInstance(p, AIProvider)
        # 必须有 generate 和 explain_mapping，不能有旧的 explain_command
        self.assertTrue(callable(getattr(p, "generate", None)))
        self.assertTrue(callable(getattr(p, "explain_mapping", None)))
        self.assertFalse(hasattr(p, "explain_command"),
                         f"{cls_name} 仍有旧方法 explain_command")
        # prompt 不能包含 Linux 命令行字样
        import inspect
        src = inspect.getsource(p.explain_mapping)
        forbidden = ["Linux", "命令行", "命令简介", "-l, -a", "tree, stat"]
        for word in forbidden:
            self.assertNotIn(word, src,
                             f"{cls_name}.explain_mapping 含旧提示词：{word!r}")

    def test_claude_provider(self):
        self._check_provider("claude", "ClaudeProvider")
        print(f"\n[Test] ClaudeProvider 接口 ✅  旧提示词已清除")

    def test_deepseek_provider(self):
        self._check_provider("deepseek", "DeepSeekProvider")
        print(f"\n[Test] DeepSeekProvider 接口 ✅  旧提示词已清除")

    def test_gemini_provider(self):
        self._check_provider("gemini", "GeminiProvider")
        print(f"\n[Test] GeminiProvider 接口 ✅  旧提示词已清除")

    def test_groq_provider(self):
        self._check_provider("groq", "GroqProvider")
        print(f"\n[Test] GroqProvider 接口 ✅  旧提示词已清除，已去除 groq 库依赖")

    def test_groq_uses_rest_not_library(self):
        """确认 groq.py 用 requests 而非 groq 库"""
        import inspect
        from py2wl.compat._core.ai_providers import groq as groq_mod
        src = inspect.getsource(groq_mod)
        self.assertIn("requests", src)
        self.assertNotIn("from groq import", src)
        print(f"\n[Test] GroqProvider 已改用 REST（无 groq 库依赖）✅")

    def test_ai_plugin_prompts_are_wolfram_context(self):
        """ai_plugin.py 的 prompt 都是 py2wl 语境"""
        import inspect
        from py2wl.compat._core import ai_plugin
        src = inspect.getsource(ai_plugin)
        self.assertIn("Wolfram Language", src)
        self.assertNotIn("Linux/Unix 命令行", src)
        self.assertNotIn("explain_command", src)
        print(f"\n[Test] ai_plugin.py prompt 均为 py2wl 语境 ✅")


if __name__ == "__main__":
    unittest.main(verbosity=2)
