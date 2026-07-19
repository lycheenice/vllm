"""Generate mkdocs.yml nav tree for the vLLM framework wiki.

Walks the wiki directory and emits an explicit ``nav:`` section that mirrors
the directory structure. Section titles are pulled from ``SECTION_TITLES``
below; pages without an explicit title inherit their first H1 from the
markdown body (standard mkdocs behavior).

Usage:
    python3 _build_nav.py > nav.yml
    # or to print the full mkdocs.yml
    python3 _build_nav.py --full
"""

from __future__ import annotations

import argparse
import pathlib
import sys

WIKI = pathlib.Path(__file__).resolve().parent

# Section titles keyed by directory path relative to the wiki root.
# Empty-string key = wiki root (used only for the home README entry).
SECTION_TITLES = {
    "00-overview": "00 · 全局资产",
    "01-engine-core": "01 · 引擎核心",
    "01-engine-core/scheduler": "调度器",
    "01-engine-core/kv-cache-management": "KV 缓存管理",
    "02-execution": "02 · 执行层",
    "02-execution/executor": "Executor",
    "02-execution/worker": "Worker",
    "03-model-execution": "03 · 模型执行",
    "03-model-execution/model-loader": "模型加载器",
    "03-model-execution/layers": "层库",
    "03-model-execution/layers/quantization": "量化层库",
    "04-model-zoo": "04 · 模型库",
    "04-model-zoo/architecture-families": "架构家族",
    "05-attention": "05 · 注意力",
    "05-attention/backends": "注意力后端",
    "05-attention/backends/mla": "MLA",
    "06-sampling-decoding": "06 · 采样与解码",
    "06-sampling-decoding/speculative-decoding": "投机解码",
    "06-sampling-decoding/structured-output": "结构化输出",
    "07-distributed": "07 · 分布式",
    "07-distributed/device-communicators": "设备通信",
    "07-distributed/kv-transfer": "KV 迁移",
    "07-distributed/kv-transfer/transports": "传输层",
    "08-platforms": "08 · 硬件平台",
    "09-compilation-ir": "09 · 编译与 IR",
    "09-compilation-ir/passes": "Inductor Pass",
    "10-config": "10 · 配置",
    "11-multimodal": "11 · 多模态",
    "12-lora": "12 · LoRA",
    "13-entrypoints": "13 · API 入口",
    "13-entrypoints/openai": "OpenAI API",
    "13-entrypoints/anthropic": "Anthropic API",
    "13-entrypoints/cli": "CLI",
    "13-entrypoints/serve": "serve 工具集",
    "13-entrypoints/pooling": "Pooling",
    "13-entrypoints/generate": "Generate",
    "13-entrypoints/scale-out": "Scale-out",
    "13-entrypoints/speech-to-text": "Speech-to-text",
    "13-entrypoints/mcp": "MCP",
    "14-tokenizers-transformers": "14 · 分词与转换器",
    "14-tokenizers-transformers/tokenizers": "分词器",
    "14-tokenizers-transformers/transformers_utils": "transformers_utils",
    "14-tokenizers-transformers/tool_parsers": "Tool Parsers",
    "15-kv-cache-offload": "15 · KV 缓存卸载",
    "16-observability": "16 · 可观测",
    "16-observability/v1/metrics": "v1 Metrics",
    "16-observability/profiler": "Profiler",
    "16-observability/tracing": "Tracing",
    "16-observability/logging_utils": "Logging Utils",
    "17-utils-cross-cutting": "17 · 工具与横切",
    "18-build-ci-testing": "18 · 构建 / CI / 测试",
    "19-appendix": "19 · 附录",
}

# Explicit ordering for top-level subsystems (so nav doesn't rely on alpha sort).
TOP_LEVEL_ORDER = [
    "00-overview",
    "01-engine-core",
    "02-execution",
    "03-model-execution",
    "04-model-zoo",
    "05-attention",
    "06-sampling-decoding",
    "07-distributed",
    "08-platforms",
    "09-compilation-ir",
    "10-config",
    "11-multimodal",
    "12-lora",
    "13-entrypoints",
    "14-tokenizers-transformers",
    "15-kv-cache-offload",
    "16-observability",
    "17-utils-cross-cutting",
    "18-build-ci-testing",
    "19-appendix",
]


def yaml_quote(s: str) -> str:
    """Quote a scalar for YAML if needed; otherwise return bare."""
    # Always quote for safety with Chinese characters and special chars.
    # Use double quotes; escape backslash and double-quote.
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def emit_section(rel_dir: str, indent: int) -> list[str]:
    """Emit YAML lines for a section (directory).

    Layout:
        <section_title>:
          - <README.md>   # section index, no title (mkdocs-material nav.indexes)
          - <subdir>:     # nested section
            - ...
          - Page: <path>  # pages get H1-derived titles unless explicit
    """
    abs_dir = WIKI / rel_dir if rel_dir else WIKI
    lines: list[str] = []
    pad = "  " * indent

    # Files first (README.md as section index without title)
    md_files = sorted(
        p for p in abs_dir.iterdir()
        if p.is_file() and p.suffix == ".md" and p.name != "README.md"
    )
    # README.md goes first (section index) when navigation.indexes is on.
    readme = abs_dir / "README.md"

    # Get subdir list, sorted alphabetically but predictable.
    sub_dirs = sorted(
        p for p in abs_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and not p.name.startswith("_")
    )

    title = SECTION_TITLES.get(rel_dir, abs_dir.name)
    lines.append(f"{pad}- {yaml_quote(title)}:")

    inner_pad = "  " * (indent + 1)

    if readme.exists():
        # Reference README as section index (no title) so mkdocs-material
        # treats it as the section page.
        readme_rel = (abs_dir / "README.md").relative_to(WIKI).as_posix()
        lines.append(f"{inner_pad}- {readme_rel}")

    # Top-level subsystem directory order is determined by TOP_LEVEL_ORDER;
    # for nested dirs we walk alphabetically.
    def sort_key(d: pathlib.Path) -> tuple:
        if not rel_dir:
            # top-level — apply TOP_LEVEL_ORDER; unknown go lexically last.
            try:
                return (0, TOP_LEVEL_ORDER.index(d.name))
            except ValueError:
                return (1, d.name)
        return (0, d.name)

    for sd in sorted(sub_dirs, key=sort_key):
        sub_rel = (sd.relative_to(WIKI)).as_posix()
        lines.extend(emit_section(sub_rel, indent + 1))

    # Finally emit non-README md files.
    for mf in md_files:
        mf_rel = mf.relative_to(WIKI).as_posix()
        lines.append(f"{inner_pad}- {mf_rel}")

    return lines


def build_nav() -> list[str]:
    lines: list[str] = ["nav:"]
    # Wiki root README + SERVE page first
    lines.append("  - README.md")
    lines.append('  - "本地预览": SERVE.md')
    # Then each top-level subsystem as a section
    for sub in TOP_LEVEL_ORDER:
        sub_dir = WIKI / sub
        if not sub_dir.is_dir():
            continue
        lines.extend(emit_section(sub, 1))
    return lines


STATIC_HEAD = """\
# mkdocs configuration for the vLLM framework wiki.
#
# This is a self-contained mkdocs project rooted at develop/wiki/.
# Serve locally:
#   pip install -r develop/wiki/requirements-mkdocs.txt
#   mkdocs serve -f develop/wiki/mkdocs.yml
#
# The nav tree below is generated from the directory structure; regenerate
# with:
#   python3 develop/wiki/_build_nav.py --full > develop/wiki/mkdocs.yml

site_name: vLLM 框架 Wiki
site_description: 系统性拆解 vLLM 框架的中文 Wiki（子系统 → 子模块 五段式）
site_url: ""
repo_url: https://github.com/lycheenice/vllm
repo_branch: v0.25.0
edit_uri: edit/v0.25.0/develop/wiki/
docs_dir: .

theme:
  name: material
  language: zh
  features:
    - navigation.instant
    - navigation.instant.progress
    - navigation.tracking
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.sections
    - navigation.indexes
    - navigation.top
    - navigation.path
    - navigation.prune
    - search.highlight
    - search.share
    - search.suggest
    - toc.follow
    - content.code.copy
    - content.code.select
    - content.tabs.link
  palette:
    - media: "(prefers-color-scheme)"
      toggle:
        icon: material/brightness-auto
        name: 自动切换主题
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: 切换至暗色
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-2
        name: 切换至亮色
  icon:
    repo: fontawesome/brands/github

plugins:
  - search:
      lang:
        - en
        - ja
  - autorefs

markdown_extensions:
  - attr_list
  - def_list
  - md_in_html
  - admonition
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.snippets
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg
  - toc:
      permalink: true
      toc_depth: 3

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/lycheenice/vllm

extra_javascript:
  - https://unpkg.com/mermaid@10/dist/mermaid.min.js
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="Emit the full mkdocs.yml, not just the nav section.")
    args = ap.parse_args()

    if args.full:
        out = [STATIC_HEAD, ""]
        out.extend(build_nav())
        out.append("")
        sys.stdout.write("\n".join(out) + "\n")
    else:
        sys.stdout.write("\n".join(build_nav()) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
