# SETUP · 环境配置与协作规范

> 面向本仓库的多人不同设备协作(CPU / GPU / 版本各异)。共享配置只放团队商定的
> 通用默认值,每台机器的差异一律走本机覆盖,不改共享文件。
>
> 文档分工:`README.md` 讲项目是什么;本文件(SETUP.md)是**配置文档**,讲怎么把
> 项目在自己机器上跑起来 —— 随仓库分发,机制变了就更新这里。

---

## 1. 快速开始

```text
双击 启动.bat
```

启动脚本自动完成:找一个能用的 Python → 查依赖(缺了自动换下一个候选)→
挑空闲端口(8788 起)→ 起服务 → 等 /health 通了自动开浏览器。

手动方式(等价):

```powershell
python scripts\launch.py            # --no-browser --port 8790 可选
```

## 2. Python 环境

**推荐给仓库建自己的 venv**(第一优先级,最稳):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

没有 `.venv` 时,启动脚本会**自动扫描仓库旁两层内的 `.venv` / `.venv-*` 目录**,
逐个检查依赖,谁全用谁 —— 不写死任何机器的项目路径。注意:借用的 venv 缺
fastapi / uvicorn / yaml / numpy / cv2 中任一项就会被跳过(看板起不来不等于环境坏,
看日志提示)。训练(torch / ultralytics)只在训练和检测接口时才需要。

## 3. 计算设备(CPU / GPU)

`config.yaml` 里一处配置,PCB 训练/推理与硅片 YOLO 训练/推理全部跟随:

```yaml
device: auto    # 有 NVIDIA GPU 用 GPU,无 GPU 回退 CPU;也可写死 cpu / 0 / 0,1
```

- 每次启动/训练开始会**打印实际用的设备**,例如:
  `[config] device: auto -> cuda:0(NVIDIA GeForce RTX 4060 Laptop GPU)`
- `auto` 想走 GPU 的前提:装的必须是 **CUDA 版 torch**(CPU 版装了也只会回退
  cpu,日志会提示原因)。安装:

  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  ```

- `/health` 接口的 `device` 字段也能看到当前解析结果。
- 想临时在 GPU 机器上强制 CPU:改 `config.local.yaml`(见下)写 `device: cpu`。

## 4. 数据放置与本地覆盖 `config.local.yaml`

共享的 `config.yaml` 默认数据在仓库外:`../分类数据/…`(团队当前约定的布局,
要改大家一起改)。
**每个人的实际数据位置可能不同,不改共享配置,建仓库根目录的
`config.local.yaml`(已被 gitignore)覆盖任意键**,与本机配置深合并、本文件优先:

```yaml
# config.local.yaml —— 本机差异:数据放仓库内(已被 .gitignore 忽略)
pcb:
  dataset_root: 分类数据/PCB分类数据
wafer:
  dataset_root: 分类数据/硅片分类数据/硅片分类数据
```

放仓库内还是仓库外都行:仓库内则依赖 `.gitignore` 的 `分类数据/` 条目;
仓库外路径随意,覆盖 `dataset_root` 指过去即可。`config.local.yaml` 里也可以
覆盖 `device`、`port` 等任意键。

## 5. 不入 git 的东西(已由 .gitignore 约定)

| 内容 | 说明 |
|------|------|
| `*.pt` / `*.pth` / 权重 | 走 GitHub Release 分发 |
| `分类数据/`、`data/**`、`runs/`、`logs/` | 数据集与训练/运行产物 |
| `config.local.yaml` | 本机差异配置 |
| 需求 PDF | 本地参考件 |

## 6. 训练与数据准备

实验与训练命令(PCB 基线 `--compare`、预标注、硅片 prepare/train 等)**以 README
「怎么跑」一节为准**,本文件不重复维护;这里只管配置与环境。

```powershell
python scripts\smoke_test.py    # 冒烟自检(不依赖权重)
```

## 7. 协作约定

1. 共享 `config.yaml` 只放**所有人通用**的默认值(团队商定);个人/单机差异进
   `config.local.yaml`。默认值要调整就在群里/PR 里说,改完提交,别人 pull 即可。
2. 代码里**不写死机器相关路径**(盘符、用户目录、某台机器的项目布局);
   需要环境信息时走配置或自动探测(参考 `scripts/launch.py` 的 venv 发现)。
3. 推送前先 `git pull --rebase origin main`;推被拒说明有人先推了新提交,
   先 `git fetch` 看差异再合并,**不要直接 `--force` 覆盖别人的提交**。
4. 数据集/权重的获取方式或配置机制变更时,更新本文件对应小节。
