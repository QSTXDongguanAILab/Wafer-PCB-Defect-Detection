# Wafer-PCB-Defect-Detection

PCB 终检假点过滤 + 光伏硅片缺陷检测。FastAPI 服务 + 模型训练/推理 + SQLite 追溯 + 缺陷处置 Agent。

本仓库只保留 PCB 与光伏两条业务线。前身 `ShopInspect` 里的钢材表面缺陷(NEU-DET 六类)相关内容
——`defect_model/`、钢材 SOP、COCO 通用检测入口——已全部去掉。

---

## 两条业务线

### PCB 假点过滤(优先开发)

产线位置:

```
PCB板 → AVI 光学检测 → 【本系统复判分选】 → 人工复检 → OK板 / NG板
```

AVI 报点里约三分之一是**假点**(误报)。把这部分自动放行掉,就是省下来的人工复检工时。

- **任务类型**:成对图像分类(不是目标检测)。AVI 每报一个疑点存一对 100×100 RGB 小图:
  `<名>.jpg` 待检图 + `<名>_T.jpg` 同位置标准模板图。
- **类别**:10 类,`假点` + 9 类真缺陷,命名为「部位(基材/焊盘) × 形态」。定义见 `pcb/labels.py`。
- **判级**:`pass` 放行 / `review` 人工复检 / `ng` 缺陷。只有高置信度假点放行,其余一律转人工。

### 光伏硅片缺陷检测

产线位置:粘晶 → 切片 → 脱胶 → 插片 → 清洗 → **分选**。对象是**硅片**,不是组件。

- **任务类型**:目标检测。640×640 灰度图,Pascal-VOC XML 标注,一张图常含多个不同代码的框。
- **类别**:12 个产线缺陷代码。**代码↔中文名对照表甲方未提供,必须向数据方索取** —— 见 `wafer/labels.py`。

---

## 汇报口径(重要)

PCB 这个任务**不能用 accuracy 汇报**,因为误判成本不对称:

| 错误方向 | 后果 | 性质 |
|---------|------|------|
| 真缺陷 → 判成假点放行 | 不良品流到客户 | 质量事故 |
| 假点 → 判成缺陷 | 多一次人工复检 | 成本 |

所以固定口径是:**NG 召回 ≥ 99%(或客户要求水位)时,能自动放行掉多少假点**。
前者是准入门槛,后者是收益。实现见 `pcb/metrics.py:operating_point`。

另一个必须守住的规矩:**训练/验证按板号分组切分**。同一块板切出的 ROI 高度相似,
随机切图会让同板样本同时进 train 和 val,指标虚高十几个点然后上线崩掉。
实现见 `pcb/dataset.py:group_split`,`tests/test_pcb_dataset.py` 有防回归测试。

---

## 目录结构

```
启动.bat             双击启动(中文名入口,内部 call run.bat)
run.bat              启动器:找 Python(ASCII 文件名+ASCII 内容,避开中文路径的代码页坑)
app/                 FastAPI 服务
  config.py          config.yaml 加载(嵌套 pcb / wafer 两段)
  db.py              SQLite 记录,两条业务线共用一张表
  tasks.py           /tasks 元信息 —— 前端类别表的唯一来源
  main.py            路由
  static/            看板前端(index.html + app.js)
pcb/                 PCB 假点过滤
  labels.py          10 类定义、别名归一、判级规则
  dataset.py         成对样本扫描、板号分组切分(纯 Python,无 torch)
  loader.py          torch Dataset、四种输入表示、同步增广
  model.py           小 CNN + 双头(二分类 + 十分类)
  metrics.py         召回优先的业务指标
  train.py           基线训练 / 输入表示 A-B
  infer.py           推理封装
wafer/               光伏硅片检测
  labels.py          12 个缺陷代码(中文名待确认)
  voc.py             VOC 解析 → YOLO 转换(含越界裁剪)
  prepare.py         数据准备 + 硅片号分组切分
  train.py / infer.py
rag_agent/           缺陷处置 RAG + Agent(SOP 已重写为 PCB 十类)
scripts/             launch / run_api / data_report / prelabel_pcb / smoke_test
tests/               pytest 单测(18 项)
```

## 环境

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
pip install -r rag_agent/requirements.txt   # 可选:缺陷处置 Agent
```

数据默认从仓库外读取,路径在 `config.yaml`:

```yaml
pcb:   dataset_root: ../分类数据/PCB分类数据
wafer: dataset_root: ../分类数据/硅片分类数据/硅片分类数据
```

> 数据位置、device(GPU/CPU)、端口等**本机差异不改上面这份共享配置** —— 在仓库根
> 目录建 `config.local.yaml`(已 gitignore)覆盖任意键;环境、GPU、本地覆盖机制的
> 完整说明见 **SETUP.md**。

---

## 怎么跑

### 体检与自测

```bash
python scripts/smoke_test.py     # 配置/数据扫描/模型前向/接口,全部不依赖权重
python scripts/data_report.py    # 两条业务线数据现状快照 → artifacts/data_report.json
python -m pytest tests -q
```

### PCB 基线实验(当前主线)

```bash
python -m pcb.train --compare              # 四种输入表示各训一遍,出对比表
python -m pcb.train --input-mode diff      # 只训一种
python -m pcb.train --epochs 2 --quick     # 冒烟,不写权重
```

输出 `artifacts/pcb/train_report.json` 与权重 `models/pcb_pair_cls.pt`(权重里存了
`input_mode` 与 `img_size`,推理端不从 config 猜,避免训练/推理表示不一致)。

`--compare` 要回答的是**「模板图有没有用、怎么用」**,这是比选 backbone 重要得多的分岔口:

| input_mode | 通道 | 含义 |
|-----------|------|------|
| `single` | 3 | 只用待检图,丢掉模板 —— 基线的基线 |
| `stack` | 6 | 待检 + 模板 通道拼接 |
| `diff` | 3 | 待检 − 模板 差分,对应 AVI 现场的判据 |
| `stack_diff` | 9 | 三者都给 |

### PCB 训练集预标注

训练集那 1149 对散图从零手分是整个项目最贵的一块人工。用基线权重先预分类:

```bash
python scripts/prelabel_pcb.py                    # 出 CSV,按不确定度升序排
python scripts/prelabel_pcb.py --move-to <目录>    # 按预测类别复制成对图
```

清单按「模型最犹豫」排前面,人工从第一行开始核对效率最高。**预标注不是真值,必须人工核对。**

### 光伏硅片

```bash
python -m wafer.prepare            # 只看概况
python -m wafer.prepare --write    # 生成 data/wafer/yolo/ + data.yaml
python -m wafer.train --epochs 100
```

### 起服务

**双击 `启动.bat`** 就行:自动找解释器 → 查依赖(缺了就换装好的那个环境)→ 端口被占就顺延 →
等 `/health` 通了自动开浏览器。真正的逻辑在 `run.bat` + `scripts/launch.py`,
`启动.bat` 只是个中文名快捷入口。

命令行等价写法:

```bash
python scripts/launch.py                       # 等同双击
python scripts/launch.py --reload              # 开发用,改代码自动重载
python scripts/launch.py --no-browser --port 8790
python scripts/run_api.py                      # 不做任何自动处理的裸启动
```

默认只监听 `127.0.0.1`。**本服务没有登录与鉴权** —— 把 `config.yaml` 的 `host` 改成
`0.0.0.0` 之前先想清楚:局域网里任何人都能上传图片、读写检测记录、删数据。
启动器检测到非本机监听会打印警告。

主要接口:

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 两条业务线的权重就绪状态 |
| GET | `/tasks` `/tasks/{pcb\|wafer}` | 类别表、接入阶段、数据现状 |
| POST | `/pcb/inspect` | 上传 `image`(+可选 `template`)→ 类别 + 假点概率 + 判级 |
| POST | `/wafer/inspect` | 上传 `image` → 缺陷框 |
| GET | `/records` `/records/{id}` `/records/export.csv` | 记录追溯与导出 |
| GET | `/stats` | KPI,含 PCB 自动放行率 |
| GET | `/agent/` `/agent/dispose?record_id=` | 缺陷处置工作台 |

权重未训练时 `/pcb/inspect`、`/wafer/inspect` 返回 **503 + 可执行提示**,不会拿未训练的模型瞎猜。

---

## 缺陷处置 Agent

`rag_agent/` 走独立配置(`rag_agent/.env`),挂在 `/agent`,langchain 栈未装时自动跳过,不影响主功能。

```bash
pip install -r rag_agent/requirements.txt
cp rag_agent/.env.example rag_agent/.env    # 填入 SILICONFLOW_API_KEY
python -m rag_agent.build_index             # 构建 SOP 向量索引
```

SOP 在 `rag_agent/data/sop/`,文件名即缺陷类名(检索按 `label` 精确过滤)。
已按 PCB 十类重写,**全部标注为「初稿,待工艺确认」** —— 里面的工艺参数和允收判据
必须经客户工艺部门确认后才能当作业指导书用。SOP 改完要重新 `build_index`。

高危动作(报废/返修/重工/停线/停机/补焊/挖修/更换/整批)由 `rag_agent/hitl` 识别,
需经 `/agent/dispose/confirm` 人工批准。

## 数据现状(2026-09-01 实测)

| | PCB | 光伏硅片 |
|---|---|---|
| 已标注 | 349 对(测试集,10 类) | 537 张 / 932 个框 |
| 未标注 | **1149 对**(训练集散图) | **510 张**(训练集无 XML) |
| 分组单位 | 403 块板 | 377 片硅片 |
| 主要缺口 | 训练集标注 | 训练集标注 + 代码中文名对照表 |

两条线的硬瓶颈都是**标注**,不是算力也不是模型选型。

## 下一步

1. 跑 `python -m pcb.train --compare`,拿到四种输入表示在「NG 召回 99% 时过滤率」上的对比。
2. 结果好 → `scripts/prelabel_pcb.py` 预标注 1149 对,人工只做纠正。
   结果差(过滤率≈0)→ 先解决输入表示或 ROI 上下文问题,这时候堆标注是白费力气。
3. 看混淆矩阵决定类别粒度:模型分不开、工艺上处置又相同的类(如基材划痕/擦花)应合并,SOP 同步合并。
4. 向数据方索取硅片缺陷代码↔中文名对照表(不阻塞 PCB,但要走沟通流程,越早越好)。
