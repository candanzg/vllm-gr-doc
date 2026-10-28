# vLLM-GR 文档使用说明

本文档说明如何在 Linux 下使用 Jekyll 管理、合并和预览 vLLM-GR 文章。

## 1. 目录结构

当前文章采用 Jekyll 的 `_posts` 和 `_includes` 目录组织：

```text
vLLM-GR低时延生成式推荐推理系统/
├── _config.yml
├── _includes/
│   └── vllm-gr/
│       ├── 01-背景与挑战.md
│       ├── 02-低时延生成式推荐推理系统vLLM-GR.md
│       ├── 03-低时延推理引起GR-Engin.md
│       ├── 04-调度与并发优化GR-Scheduler.md
│       ├── 05-池化加速GR-Cache.md
│       ├── 06-竞品性能分析.md
│       └── 07-总结与展望.md
├── _posts/
│   └── 2026-10-27-vllm-gr.md
├── assets/
└── Jekyll使用说明.md
```

`_includes/vllm-gr/` 下的文件是分章节编辑文件。  
`_posts/2026-10-27-vllm-gr.md` 是文章入口文件，负责定义文章信息和章节顺序。

Jekyll 构建时，会按照入口文件中的 Liquid `include` 指令，把章节组合成一篇完整网页。

## 2. 安装 Linux 环境

以 Ubuntu 或 Debian 为例：

```bash
sudo apt-get update
sudo apt-get install -y ruby-full build-essential
gem install --user-install bundler jekyll
```

如果系统找不到 `bundle` 或 `jekyll`，执行：

```bash
GEM_BIN="$(ruby -r rubygems -e 'puts Gem.user_dir')/bin"
export PATH="$GEM_BIN:$PATH"
```

检查版本：

```bash
ruby --version
bundle --version
jekyll --version
```

如果项目中没有 `Gemfile`，可以在根目录创建：

```ruby
source "https://rubygems.org"

gem "jekyll", "~> 4.3"
gem "minima", "~> 2.5"
gem "webrick"
```

然后安装依赖：

```bash
bundle install
```

如果使用 vLLM 官方博客仓库，应优先使用仓库已有的 `Gemfile`。

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
{% include vllm-gr/01-背景与挑战.md %}
{% include vllm-gr/02-低时延生成式推荐推理系统vLLM-GR.md %}
{% include vllm-gr/03-低时延推理引起GR-Engin.md %}
{% include vllm-gr/04-调度与并发优化GR-Scheduler.md %}
{% include vllm-gr/05-池化加速GR-Cache.md %}
{% include vllm-gr/06-竞品性能分析.md %}
{% include vllm-gr/07-总结与展望.md %}
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
git diff -- "_includes/vllm-gr/03-低时延推理引起GR-Engin.md"
```

构建通过后提交：

```bash
git add "_includes/vllm-gr/03-低时延推理引起GR-Engin.md"
git commit -m "docs: update GR engine chapter"
git push -u origin docs/gr-chapter-03
```

章节 PR 合并到主分支后，Jekyll 会在构建时自动把各章节组合成完整文章。

