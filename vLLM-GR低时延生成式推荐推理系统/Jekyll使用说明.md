# vLLM-GR 文档使用说明

本文档说明如何在 Linux 下使用 Jekyll 管理、合并和预览 vLLM-GR 文章。

## 1. 目录结构

当前文章采用 Jekyll 的 `_posts` 和 `_includes` 目录组织：

```text
vLLM-GR低时延生成式推荐推理系统/
├── Gemfile
├── _config.yml
├── _includes/
│   └── vllm-gr/
│       ├── 01-background-and-challenges.md
│       ├── 02-system-overview.md
│       ├── 03-gr-engine.md
│       ├── 04-gr-scheduler.md
│       ├── 05-gr-cache.md
│       ├── 06-competitive-analysis.md
│       └── 07-summary-and-outlook.md
├── _posts/
│   └── 2026-10-27-vllm-gr.md
├── assets/
├── performance_data/
└── Jekyll使用说明.md
```

`_includes/vllm-gr/` 下的文件是分章节编辑文件。  
`_posts/2026-10-27-vllm-gr.md` 是文章入口文件，负责定义文章信息和章节顺序。

> **文件名必须使用 ASCII 字符。** Jekyll 4.4 的 `include` 标签会校验文件名，
> 中文（或其他非 ASCII）路径会直接报错
> `Invalid syntax for include tag. File contains invalid characters or sequences`。
> 因此章节文件统一用英文短名，中文章节标题只写在文件正文的 `#` 标题里。

Jekyll 构建时，会按照入口文件中的 Liquid `include` 指令，把章节组合成一篇完整网页。

## 2. 安装 Linux 环境

以 Ubuntu 或 Debian 为例：

```bash
sudo apt-get update
sudo apt-get install -y ruby-full build-essential

# 安装 Jekyll 及其依赖（gem 默认装到系统目录，需要 root）
gem install 'jekyll:~>4.4' 'minima:~>2.5' webrick --no-document
```

检查版本：

```bash
ruby --version      # 需要 2.7 以上，本项目使用 3.2
jekyll --version    # 需要 4.x
```

Ubuntu/Debian 用 apt 安装的 Ruby 只提供带版本号的可执行文件（`bundle3.2`、
`bundler3.2`），没有 `bundle` 这个短名。如果 `bundle --version` 报 command not found，
就直接用带版本号的命令，下文出现的 `bundle` 都按这个规则替换：

```bash
bundle3.2 --version
```

本项目根目录已经带有 `Gemfile`：

```ruby
source "https://rubygems.org"

gem "jekyll", "~> 4.4"
gem "minima", "~> 2.5"
gem "webrick", "~> 1.9"
```

首次进入项目时安装依赖：

```bash
bundle install
```

`Gemfile.lock` 需要一起提交，保证每个人构建出的页面一致。

## 3. 进入项目目录

Linux 使用 `/` 分隔路径。例如：

```bash
PROJECT_DIR="/opt/docs/vLLM-GR低时延生成式推荐推理系统"
cd "$PROJECT_DIR"
```

检查章节文件：

```bash
find "_includes/vllm-gr" -maxdepth 1 -type f -name "*.md" -print | sort
```

## 4. 多人分章节编辑

每个人可以负责一个章节。章节文件只保留标题和正文，不要添加 YAML front matter：

```markdown
# 3 低时延推理引起GR-Engin

## 3.1 Beam Search 优化

正文内容……
```

文章级别的 `title`、`date`、`summary` 和 `tags` 放在：

```text
_posts/2026-10-27-vllm-gr.md
```

文章顺序由入口文件中的 `include` 顺序控制：

```liquid
{% include vllm-gr/01-background-and-challenges.md %}
{% include vllm-gr/02-system-overview.md %}
{% include vllm-gr/03-gr-engine.md %}
{% include vllm-gr/04-gr-scheduler.md %}
{% include vllm-gr/05-gr-cache.md %}
{% include vllm-gr/06-competitive-analysis.md %}
{% include vllm-gr/07-summary-and-outlook.md %}
```

Linux 文件名区分大小写，`include` 中的路径必须和实际文件名完全一致。

## 5. 图片和资源

图片建议放到：

```text
assets/figures/vllm-gr/
```

Markdown 引用示例：

```markdown
![总体架构](/assets/figures/vllm-gr/overall-architecture.png)
```

如果网站设置了 `baseurl`，可以使用：

```liquid
![总体架构]({{ '/assets/figures/vllm-gr/overall-architecture.png' | relative_url }})
```

## 6. 构建合并后的文章

在项目根目录运行：

```bash
bundle exec jekyll build --trace
```

Jekyll 会读取 `_posts/2026-10-27-vllm-gr.md`，解析所有 `include`，并把结果输出到：

```text
_site/
```

这里的“合并”是构建成完整 HTML 页面，不会自动生成新的合并 Markdown 文件。

清理后重新构建：

```bash
bundle exec jekyll clean
bundle exec jekyll build --trace
```

`jekyll clean` 不会删除 `_posts/`、`_includes/`、`assets/` 或章节内容。

## 7. 本地预览

启动服务：

```bash
bundle exec jekyll serve --host 0.0.0.0 --port 4000
```

浏览器访问：

```text
http://127.0.0.1:4000/
```

如果文章日期晚于系统日期，使用：

```bash
bundle exec jekyll serve --host 0.0.0.0 --port 4000 --future
```

远程 Linux 服务器可使用 SSH 端口转发：

```bash
ssh -L 4000:127.0.0.1:4000 user@your-server
```

然后在本地访问 `http://127.0.0.1:4000/`。

## 8. 发布前检查

```bash
test -f "_posts/2026-10-27-vllm-gr.md" && echo "post: OK"
find "_includes/vllm-gr" -maxdepth 1 -type f -name "*.md" -print | sort
bundle exec jekyll build --trace
```

常见问题：

- `_posts/2026-10-27-vllm-gr.md` 必须位于 `_posts/` 目录；
- front matter 开头和结尾的 `---` 必须完整；
- `include` 路径必须和实际文件名完全一致；
- 图片路径和文件名区分大小写；
- 代码块必须正确闭合；
- 修改依赖后要重新执行 `bundle install`。

## 9. Git 协作

每个人在自己的分支编辑章节：

```bash
git switch -c docs/gr-chapter-03
```

检查改动：

```bash
git status --short
git diff -- "_includes/vllm-gr/03-gr-engine.md"
```

构建通过后提交：

```bash
git add "_includes/vllm-gr/03-gr-engine.md"
git commit -m "docs: update GR engine chapter"
git push -u origin docs/gr-chapter-03
```

章节 PR 合并到主分支后，Jekyll 会在构建时自动把各章节组合成完整文章。

