# 本地预览 Wiki（mkdocs）

[← Wiki 首页](README.md) > 本地预览

本 wiki 自带一份 mkdocs 配置（`mkdocs.yml`），可本地起站点浏览。所有 mermaid 图、面包屑导航、二级侧栏目录均可正常渲染。

## 1. 安装依赖

按 vLLM `AGENTS.md` 规定，所有 Python 命令走 `uv` + `.venv`：

```bash
uv pip install -r develop/wiki/requirements-mkdocs.txt
```

依赖清单（见 [`requirements-mkdocs.txt`](requirements-mkdocs.txt)）：
- `mkdocs>=1.6`
- `mkdocs-material>=9.5`
- `pymdown-extensions>=10.0`（提供 superfences / mermaid fence）
- `pygments>=2.15`

## 2. 启动开发服务器

```bash
mkdocs serve -f develop/wiki/mkdocs.yml
```

默认地址 `http://127.0.0.1:8000`。文件改动会热重载。

首次构建前可校验配置：
```bash
mkdocs build -f develop/wiki/mkdocs.yml --strict
```

`--strict` 会把所有 broken link / 缺失资源标为错误，是审校 wiki 的好工具。

## 3. 构建静态站点

```bash
mkdocs build -f develop/wiki/mkdocs.yml --site-dir develop/wiki/site
```

产物在 `develop/wiki/site/`，可直接 `python3 -m http.server -d develop/wiki/site 8000` 预览或上传任意静态托管。

## 4. 重新生成 nav

`mkdocs.yml` 的 `nav:` 段由 [`_build_nav.py`](_build_nav.py) 按目录结构自动生成。新增/重命名子模块后需重跑：

```bash
python3 develop/wiki/_build_nav.py --full > develop/wiki/mkdocs.yml
```

新增顶层子系统时同步更新 `_build_nav.py` 中的 `SECTION_TITLES` 与 `TOP_LEVEL_ORDER`，以及顶层 [`README.md`](README.md) 的导航树。

## 5. 配置要点

- 主题：`mkdocs-material`，含 light/dark/auto 三态切换、`navigation.tabs.sticky`、`navigation.indexes`（README 作为子系统首页并入侧栏根节点）、`navigation.path`（面包屑）。
- mermaid：通过 `pymdownx.superfences` 的 `custom_fences` 渲染，并额外加载 `mermaid@10` 的 JS（`extra_javascript`）。
- 搜索：`search` 插件配 `lang: [en, ja]`（mkdocs 无 zh 内置分词器，ja 是最接近的 CJK 切词）。
- edit_uri 指向 `lycheenice/vllm` fork 的 `v0.25.0` 分支，每个页面右上角"编辑"图标直跳 GitHub。

## 6. 与主仓库 docs 的关系

- 主仓库的 `mkdocs.yaml`（仓库根）服务 `docs/` 下的用户文档，与本 wiki **完全独立**。本 wiki 不写入主 mkdocs 站点，仅作为内部代码剖析资料。
- 仓库根 `.readthedocs.yaml` 不包含 `develop/wiki/`，因此 Read the Docs 不会自动构建本 wiki。

[← 返回 Wiki 首页](README.md)
