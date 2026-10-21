# Git 提交流程 - hub-ZBbE 作业仓库

> 创建时间：2026-10-21 11:11
> 更新时间：2026-10-21 11:28
> 生成模型：DeepSeek-V4.1-Flash
> 调用技能：无
> 生成时间：2026-10-21 11:11

## 一. 仓库关系

作业最终合并到老师仓库 `nlp-lwMZL/hub-ZBbE`，但个人不能直接推送到它，只能推送到自己的 fork，再通过 Pull Request (PR) 由老师审批合并。

| 角色 | 地址 | 说明 |
| --- | --- | --- |
| 本地工作区 | `/www/wwwroot/default/badou/hub-ZBbE` | 改文件、提交都在这里 |
| origin (个人 fork) | https://github.com/simonZD/hub-ZBbE.git | 只作为中转站, push 目标 |
| upstream (老师仓库) | https://github.com/nlp-lwMZL/hub-ZBbE.git | 只读同步源, 通过 PR 写入 |

数据流: 本地 `commit` -> `push origin` -> 在 GitHub 发 PR (base 为 `nlp-lwMZL:main`) -> 老师审批合并 -> 再从 `upstream` 同步回本地.

## 二. 当前实际状态

以下为 2026-10-21 11:11 实际核对结果, 与本机不一致时以命令输出为准.

- 本地分支 `main`, HEAD 为 `05010ea` (新增 .gitignore 并上传 week02 模型文件).
- `origin` 已确认是 `nlp-lwMZL/hub-ZBbE` 的 fork (`fork: true`, parent 即老师仓库).
- 老师仓库 `main` 停在 `061abb4`, 本地 `main` 领先 7 个提交, 说明这些提交还未合并进老师仓库.
- 本机尚未配置 `upstream` 远程, 需要按第三节补一次.
- 凭证方式为 `credential.helper = store`, 密码以明文存在 `~/.git-credentials`, 推送时不会重复提示, 注意不要外传该文件.
- `.codebuddy/`, `__pycache__/`, `**.bin`, `*.log` 等已被 `.gitignore` 忽略; `.vscode/` 由 `.git/info/exclude` 本地忽略 (两者都不会进入仓库).
- `张冬/week02/model.bin` 目前没有被跟踪 (被 `**.bin` 规则拦截), 若作业要求提交模型文件, 必须用 `git add -f` 强制添加.

## 三. 一次性准备

每条命令前的文字说明为目的, 只在首次配置时执行.

配置老师仓库为 upstream, 用于同步原仓库的更新.
```bash
git remote add upstream https://github.com/nlp-lwMZL/hub-ZBbE.git
```

确认两个远程都指向正确地址 (origin 为个人 fork, upstream 为老师仓库).
```bash
git remote -v
```

关闭路径转义, 中文与空格文件名在 `git status` 中正常显示.
```bash
git config core.quotepath false
```

确认提交身份为本人 GitHub 账号与邮箱 (与 fork 账号一致).
```bash
git config user.name
git config user.email
```

拉取老师仓库最新提交, 之后发 PR 前都要先做这一步.
```bash
git fetch upstream
```

凭据说明: GitHub 已不支持账号密码推送, 需使用 Personal Access Token (权限勾选 `repo`), 首次推送时用户名填 GitHub 账号、密码填 token, 由 `store` 记住.

## 四. 每次提交作业的标准流程

有两种方式, **推荐方式 B**, 它能让每次 PR 只包含本周作业.

### 方式 A: 直接在 main 上提交 - 目前用这个 (简单, 但 PR 会带上历史累计提交)

同步老师仓库最新代码, 避免与自己改动冲突. `git pull upstream main` 等价于 `git fetch upstream` + `git merge upstream/main`, 但只作用于当前分支, 所以切换分支不能省.
```bash
git switch main
git pull upstream main
```

建立本周目录, 目录名与周次严格按 README 规范 (姓名/weekNN).
```bash
mkdir -p 张冬/week03
```

把作业文件放进该目录后, 查看将要提交的内容, 确认只有自己的文件.
```bash
git status
git diff --stat
```

暂存本周目录, 只加自己名下路径, 不要 `git add .`.
```bash
git add 张冬/week03
```

需要提交被忽略的模型或数据文件时, 单独强制添加.
```bash
git add -f 张冬/week03/model.bin
```

提交, 提交信息写清姓名与周次.
```bash
git commit -m "张冬-week03 作业"
```

推送到自己的 fork.
```bash
git push origin main
```

打开 PR: 浏览器访问下面地址, base 选 `nlp-lwMZL/hub-ZBbE` 的 `main`, compare 选 `simonZD/hub-ZBbE` 的 `main`.
```text
https://github.com/nlp-lwMZL/hub-ZBbE/compare/main...simonZD:main?expand=1
```

填写 PR 标题 (如 `张冬-week03 作业`) 与说明, 提交后等待老师审批, 不要反复关闭重开.

### 方式 B: 每周单独分支 (推荐, PR 只含本周内容)

先同步老师仓库, 再以它的最新提交为起点建分支, 分支名用英文避免编码问题.
```bash
git fetch upstream
git switch -c zdong-week03 upstream/main
```

放入作业并只暂存自己的目录.
```bash
git add 张冬/week03
```

提交并推送到 fork, 同时建立追踪关系.
```bash
git commit -m "张冬-week03 作业"
git push -u origin zdong-week03
```

发 PR 时 base 为 `nlp-lwMZL:main`, compare 为 `simonZD:zdong-week03`, 这样 PR 中只有本周改动.

合并通过后清理本地分支, 并让 main 跟上老师仓库.
```bash
git switch main
git pull upstream main
git branch -d zdong-week03
```

## 五. 提交规范

- 目录: 只在自己的姓名目录下建 `weekNN` 子目录, 禁止改动其他同学与老师的文件, 禁止删除他人目录.
- 命名: 文件名不得含空格与半角点号, 用 `-` 或 `_` 连接 (现有 `张冬/week01/ai工具-codebuddy 已安装.jpg` 属历史文件, 新文件不要再这样命名).
- 提交信息: 统一 `姓名-weekNN 作业` 或 `姓名-weekNN 作业说明`, 一次提交只做一件事, 不使用 `update`、`test` 这类无信息量的信息.
- 不提交产物: `__pycache__/`、`*.log`、临时文件、虚拟环境一律不入库, 已被 `.gitignore` 拦截, 不要用 `-f` 绕过.
- 大文件: 模型与数据超过 50MB 时 GitHub 会拒收, 优先提交代码与说明, 必须交模型时先确认体积.

## 六. 常见问题处理

推送被拒 (`non-fast-forward`), 说明 fork 上已有自己没拉的提交, 先拉取再推送.
```bash
git pull --rebase origin main
git push origin main
```

误把缓存或大文件加进暂存区, 只从 Git 索引移除, 不删本地文件.
```bash
git rm -r --cached 张冬/week03/__pycache__
git commit -m "移除误提交的缓存文件"
```

撤销最近一次还未推送的提交, 改动保留在工作区.
```bash
git reset --soft HEAD~1
```

修改上一次提交信息 (未推送时).
```bash
git commit --amend -m "张冬-week03 作业"
```

PR 提示冲突, 把分支变基到老师仓库最新代码上, 解决冲突后再推送.
```bash
git fetch upstream
git rebase upstream/main
git push --force-with-lease origin zdong-week03
```

查看某次提交动了哪些文件, 用于确认自己没碰别人的目录.
```bash
git show --stat 05010ea
```

确认某文件为什么没被提交 (是否被忽略).
```bash
git check-ignore -v 张冬/week02/model.bin
```

## 七. 提交前自检清单

- [ ] 目录路径为 `姓名/weekNN`, 周次编号正确.
- [ ] `git status` 中只出现自己名下的新增或修改文件.
- [ ] 没有 `__pycache__`、日志、虚拟环境等产物.
- [ ] 文件名不含空格与半角点号.
- [ ] 提交信息包含姓名与周次.
- [ ] 推送目标是 `origin` 的自己分支, 没有直接推 `upstream`.
- [ ] PR 的 base 是 `nlp-lwMZL/hub-ZBbE` 的 `main`.
