# CURRENT_PROGRESS · Wafer-PCB-Defect-Detection

> 更新:2026-09-01(Asia/Shanghai)
> 用途:跨会话续作真源。新对话先读本文件,不要重做已完成项。

## 一句话状态

新仓库骨架已搭好并跑通(smoke 8/8、pytest 18/18、训练与数据准备链路实测可跑)。
**PCB 与光伏两条线的模型都还没训练** —— 卡在标注,不在代码。

## 已完成(勿重复)

- 从 `AI视觉/ShopInspect` 剥离出只含 PCB + 光伏的新仓库,钢材(NEU-DET 六类)相关内容全部去掉:
  `defect_model/`、钢材 SOP、COCO 通用检测入口、camera 采集、MES/ZIP 导出。
- `app/`:配置(嵌套 pcb/wafer)、SQLite 记录(task 字段区分业务线)、`/tasks` 元信息、
  `/pcb/inspect`、`/wafer/inspect`、记录追溯与 CSV 导出、`/stats`(含 PCB 自动放行率)。
- `pcb/`:10 类定义与别名归一、成对样本扫描、**板号分组切分**、四种输入表示、
  小 CNN 双头模型、召回优先指标、基线训练(`--compare` 做输入表示 A/B)、推理封装。
- `wafer/`:12 个缺陷代码、VOC 解析与 YOLO 转换(含越界裁剪与稀有类剔除)、
  **硅片号分组切分**、数据准备与训练脚本。
- `app/static/`:看板前端。类别表由 `/tasks` 驱动,前端不写死类名。
- `rag_agent/`:代码整体复用;SOP 重写为 PCB 十类(11 篇,均标「待工艺确认」);
  高危词表、SYSTEM_PROMPT、`top_label→label` 字段适配已改完。
- `scripts/`:`run_api` / `data_report` / `prelabel_pcb` / `smoke_test`。
- `tests/`:18 项单测,覆盖文件名解析、分组切分无泄漏、判级不对称、指标口径、VOC 越界裁剪。

## 明确未做

- PCB 基线训练未跑(只做过 2 epoch 冒烟)。`models/pcb_pair_cls.pt` 不存在,
  `/pcb/inspect` 目前返回 503。
- PCB 训练集 1149 对未标注。
- 硅片模型未训练;`wafer.prepare --write` 未落盘;代码↔中文名对照表未拿到。
- RAG 向量索引未在本仓库构建(`python -m rag_agent.build_index` 待跑)。
- 未推送到 GitHub 远端(会覆盖 `QSTXDongguanAILab/Wafer-PCB-Defect-Detection` 现有内容,需人工确认)。
- 旧仓库 `AI视觉/ShopInspect` 原样保留,未删除。

## 环境备注

当前用的是 `AI视觉/ShopInspect/.venv`(torch 2.13+cpu、ultralytics 8.4、fastapi 0.141、
langchain 1.3、chromadb 1.5、pytest 9.1 都在)。本仓库自建 venv 时按 `requirements.txt` 装。

CPU 训练耗时参考:2 epoch × 256 训练样本 ≈ 42 秒,所以 30 epoch 单模式约 10 分钟,
`--compare` 四种模式约 40 分钟。

## 建议下一刀

1. `python -m pcb.train --compare` → 看四种输入表示在「NG 召回 99% 时过滤率」上的差距。
2. 按结果决定是预标注(结果好)还是改输入表示/ROI 上下文(结果差)。
3. 看混淆矩阵合并分不开且处置相同的类,SOP 同步合并。
4. 向数据方索取硅片缺陷代码对照表。

## 新会话口令

```text
读 G:\PCB与光伏项目\Wafer-PCB-Defect-Detection\CURRENT_PROGRESS.md,按「建议下一刀」继续,不要重做已完成项。
```
