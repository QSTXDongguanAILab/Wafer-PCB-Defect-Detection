/* PCB / 光伏硅片缺陷检测 —— 看板前端。
   页面结构全部由 JS 依据 /tasks 渲染:类别表只在后端定义一次,
   避免出现「前端写死一套类名、后端另一套」的错位。 */
"use strict";

var state = { health: null, tasks: [], stats: null, page: "home" };
var pcbFiles = { test: null, template: null };
var waferFile = null;

function el(id) { return document.getElementById(id); }

function esc(s) {
  return String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function pct(v) { return v === null || v === undefined ? "—" : (v * 100).toFixed(1) + "%"; }

/* 后端未捕获的异常会返回 text/plain "Internal Server Error",
   直接 r.json() 会抛 SyntaxError 把真实错误吞掉。统一走这里。 */
async function getJSON(url, opts) {
  var r = await fetch(url, opts);
  var text = await r.text();
  var data = null;
  try { data = text ? JSON.parse(text) : null; } catch (_) {}
  if (!r.ok) {
    var msg = data && data.detail ? data.detail : (text || "").slice(0, 400);
    throw new Error("HTTP " + r.status + " · " + (msg || r.statusText));
  }
  if (data === null) throw new Error("响应不是 JSON:" + (text || "").slice(0, 300));
  return data;
}

function showMsg(containerId, text, kind) {
  var box = el(containerId);
  if (!box) return;
  box.className = "msg " + (kind || "err");
  box.innerHTML = esc(text);
  box.classList.remove("hidden");
}

function statusTag(status) {
  var map = { pass: "放行", review: "人工复检", ng: "缺陷 NG" };
  if (!status) return "—";
  return '<span class="tag ' + esc(status) + '">' + esc(map[status] || status) + "</span>";
}

function taskOf(name) {
  for (var i = 0; i < state.tasks.length; i++) if (state.tasks[i].task === name) return state.tasks[i];
  return null;
}

function renderKpis() {
  var s = state.stats || {};
  var pcb = taskOf("pcb"), wafer = taskOf("wafer");
  var pcbData = (pcb && pcb.dataset) || {}, waferData = (wafer && wafer.dataset) || {};
  var items;
  if (state.page === "pcb") {
    items = [
      [pcbData.labeled || 0, "已标注样本对"],
      [pcbData.unlabeled || 0, "待标注样本对"],
      [pcbData.boards || 0, "涉及板号"],
      [s.pcb_records || 0, "已检记录"],
      [pct(s.pcb_release_rate), "自动放行率"]
    ];
  } else if (state.page === "wafer") {
    items = [
      [waferData.xml_files || 0, "已标注图"],
      [waferData.boxes || 0, "缺陷框"],
      [waferData.unlabeled_images || 0, "待标注图"],
      [waferData.wafers || 0, "涉及硅片"]
    ];
  } else {
    items = [
      [s.total_records || 0, "检测记录"],
      [(s.by_task && s.by_task.pcb) || 0, "PCB 记录"],
      [(s.by_task && s.by_task.wafer) || 0, "硅片记录"],
      [pct(s.pcb_release_rate), "PCB 自动放行率"],
      [s.avg_elapsed_ms === null || s.avg_elapsed_ms === undefined ? "—" : s.avg_elapsed_ms + " ms", "平均推理耗时"]
    ];
  }
  el("kpiRow").innerHTML = items.map(function (kv) {
    return '<div class="kpi"><b>' + esc(kv[0]) + "</b><span>" + esc(kv[1]) + "</span></div>";
  }).join("");
}

function classChips(task) {
  return task.classes.map(function (c) {
    var cls = c.is_ok ? "chip ok" : (c.note === "样本个位数,暂不参与训练" ? "chip rare" : "chip");
    var note = c.note ? ' <i>' + esc(c.note) + "</i>" : "";
    return '<span class="' + cls + '">' + esc(c.id) + " · " + esc(c.name) + note + "</span>";
  }).join("");
}

function notesList(task) {
  return '<ul class="notes">' + task.notes.map(function (n) { return "<li>" + esc(n) + "</li>"; }).join("") + "</ul>";
}

function pageHome() {
  var h = state.health || {};
  var cards = state.tasks.map(function (t) {
    var d = t.dataset || {};
    var ready = t.ready
      ? '<span class="pill ok">权重已就绪</span>'
      : '<span class="pill wip">模型未训练</span>';
    var dataLine = t.task === "pcb"
      ? "已标注 " + (d.labeled || 0) + " 对 / 待标注 " + (d.unlabeled || 0) + " 对"
      : "已标注 " + (d.xml_files || 0) + " 张 / 待标注 " + (d.unlabeled_images || 0) + " 张";
    return '<div class="card"><h2>' + esc(t.title) + " " + ready +
      '<span class="pill todo">' + esc(t.kind === "classification" ? "成对分类" : "目标检测") + "</span></h2>" +
      '<p class="hint">当前阶段:' + esc(t.stage) + " · " + esc(dataLine) +
      " · 权重路径 <code>" + esc(t.model_path) + "</code></p>" +
      '<h3>缺陷类别（' + t.classes.length + " 类）</h3><div class=\"chips\">" + classChips(t) + "</div>" +
      '<div class="toolbar"><button class="btn btn-primary" data-goto="' + esc(t.task) + '">进入' +
      esc(t.task === "pcb" ? "PCB" : "光伏") + "工作台</button></div></div>";
  }).join("");

  return cards +
    '<div class="card"><h2>系统状态</h2>' +
    '<p class="hint">设备 <code>' + esc(h.device || "-") + "</code> · 版本 " + esc(h.version || "-") +
    " · 处置 Agent " + (h.agent_enabled ? "已启用" : "未启用") + "</p>" +
    '<div class="box info"><h4>这套系统在产线上的位置</h4>' +
    "PCB:PCB板 → AVI 设备 → <b>本系统复判分选</b> → 人工复检 → OK/NG。AVI 报点里约三分之一是假点," +
    "把这部分自动放行掉,就是省下来的人工复检工时。<br>" +
    "光伏:粘晶 → 切片 → 脱胶 → 插片 → 清洗 → <b>分选</b>,对硅片(不是组件)做缺陷检测。</div></div>";
}

function pagePcb() {
  var t = taskOf("pcb");
  if (!t) return '<div class="card">任务信息加载失败</div>';
  var d = t.dataset || {};
  var stages = [
    ["done", "数据打通", "成对 ROI 扫描、板号分组切分、别名归一都已就绪(scripts/data_report.py)"],
    [t.ready ? "done" : "doing", "基线实验", "四种输入表示 A/B:<code>python -m pcb.train --compare</code>,选模标准是 NG 召回 ≥99% 时的假点过滤率"],
    [d.unlabeled ? "" : "done", "训练集标注", "还有 " + (d.unlabeled || 0) + " 对未分类。先用基线权重跑 <code>scripts/prelabel_pcb.py</code> 预标注,人工只做纠正"],
    ["", "接入产线", "阈值定档 → 接 AVI 输出目录 → 记录落库 → 处置 SOP 联动"]
  ];
  var pill = t.ready ? '<span class="pill ok">权重已就绪</span>' : '<span class="pill wip">模型未训练</span>';

  return '<div class="card"><h2>' + esc(t.title) + " " + pill + "</h2>" +
    '<p class="hint">AVI 每报一个疑点会存一对 100×100 小图:待检图 + 同位置标准模板图（<code>_T</code>）。' +
    "本页对单个疑点做复判,给出类别、假点概率与放行判级。</p>" +
    '<div class="grid2">' +
      '<div><div class="drop" id="dropTest"><b>待检图</b><br>点击或拖入 <code>xxx.jpg</code></div></div>' +
      '<div><div class="drop" id="dropTmpl"><b>模板图（可选）</b><br>点击或拖入 <code>xxx_T.jpg</code></div></div>' +
    "</div>" +
    '<input type="file" id="filePcbTest" accept="image/*" class="hidden" />' +
    '<input type="file" id="filePcbTmpl" accept="image/*" class="hidden" />' +
    '<div class="toolbar">' +
      '<label class="f">工单<input type="text" id="pcbWo" placeholder="WO-2026-001" size="14" /></label>' +
      '<label class="f">板号<input type="text" id="pcbBoard" placeholder="可留空" size="16" /></label>' +
      '<label class="f">AVI 检测项<input type="text" id="pcbItem" placeholder="线路 / 大焊盘均匀度" size="16" /></label>' +
      '<button class="btn btn-primary" id="btnPcbRun">复判</button>' +
      '<button class="btn" id="btnPcbClear">清空</button>' +
    "</div>" +
    '<div id="pcbMsg" class="msg hidden"></div>' +
    '<div id="pcbResult"></div></div>' +

    '<div class="card"><h2>缺陷类别（' + t.classes.length + " 类）</h2>" +
    '<p class="hint">命名规律:部位（基材 / 焊盘）× 缺陷形态。绿色那个是放行类。</p>' +
    '<div class="chips">' + classChips(t) + "</div>" +
    '<h3>接入进度</h3><ul class="stages">' +
    stages.map(function (s) {
      return '<li class="' + s[0] + '"><b>' + esc(s[1]) + "</b><em>" + s[2] + "</em></li>";
    }).join("") + "</ul>" +
    '<h3>关键约定</h3>' + notesList(t) +
    '<div class="box warn" style="margin-top:12px"><h4>数据现状</h4>' +
    "已标注 <b>" + (d.labeled || 0) + "</b> 对(来自测试集,可直接训基线)," +
    "未标注 <b>" + (d.unlabeled || 0) + "</b> 对,缺模板图 " + (d.missing_template || 0) + " 对,涉及 " +
    (d.boards || 0) + " 块板。<br>标注是当前唯一的硬瓶颈 —— 不是算力,也不是模型选型。</div></div>";
}

function pageWafer() {
  var t = taskOf("wafer");
  if (!t) return '<div class="card">任务信息加载失败</div>';
  var d = t.dataset || {};
  var pill = t.ready ? '<span class="pill ok">权重已就绪</span>' : '<span class="pill wip">数据准备中</span>';
  return '<div class="card"><h2>' + esc(t.title) + " " + pill + "</h2>" +
    '<p class="hint">640×640 灰度硅片图,目标检测任务(不是分类)。标注为 Pascal-VOC XML,一张图常含多个不同代码的框。</p>' +
    '<div class="drop" id="dropWafer"><b>硅片图</b><br>点击或拖入 <code>xxx.png</code></div>' +
    '<input type="file" id="fileWafer" accept="image/*" class="hidden" />' +
    '<div class="toolbar">' +
      '<label class="f">工单<input type="text" id="waferWo" placeholder="WO-2026-001" size="14" /></label>' +
      '<button class="btn btn-primary" id="btnWaferRun">检测</button>' +
      '<button class="btn" id="btnWaferClear">清空</button>' +
    "</div>" +
    '<div id="waferMsg" class="msg hidden"></div><div id="waferResult"></div></div>' +

    '<div class="card"><h2>缺陷代码（' + t.classes.length + " 类）</h2>" +
    '<div class="chips">' + classChips(t) + "</div>" +
    '<div class="box warn" style="margin-top:12px"><h4>代码↔中文名对照表缺失</h4>' +
    "需求说明里没给对照表,虚线框的三类样本还只有个位数。这张表必须向数据方索取," +
    "拿到前不要自行猜译 —— 一旦猜错,看板文案和处置 SOP 会整套跟着错。</div>" +
    '<h3>数据现状</h3><p class="hint">已标注 <b>' + (d.xml_files || 0) + "</b> 张 / " +
    (d.boxes || 0) + " 个框 / " + (d.wafers || 0) + " 片硅片;训练集另有 <b>" +
    (d.unlabeled_images || 0) + "</b> 张没有 XML,现阶段用不上。</p>" +
    '<h3>关键约定</h3>' + notesList(t) +
    '<div class="box info" style="margin-top:12px"><h4>准备命令</h4>' +
    "<code>python -m wafer.prepare</code> 看概况 → <code>python -m wafer.prepare --write</code> 生成 YOLO 目录 → " +
    "<code>python -m wafer.train</code> 训练。切分按硅片号分组,同片多图不跨 train/val。</div></div>";
}

function pageRecords() {
  return '<div class="card"><h2>检测记录</h2>' +
    '<div class="toolbar">' +
      '<label class="f">业务线<select id="fTask"><option value="">全部</option><option value="pcb">PCB</option><option value="wafer">光伏硅片</option></select></label>' +
      '<label class="f">判级<select id="fStatus"><option value="">全部</option><option value="pass">放行</option><option value="review">人工复检</option><option value="ng">缺陷 NG</option></select></label>' +
      '<label class="f">工单<input type="text" id="fWo" size="14" /></label>' +
      '<button class="btn" id="btnQuery">查询</button>' +
      '<a class="btn" id="lnkCsv" href="/records/export.csv">导出 CSV</a>' +
    "</div>" +
    '<div id="recMsg" class="msg hidden"></div>' +
    '<div style="overflow:auto;margin-top:10px"><table><thead><tr>' +
    "<th>ID</th><th>时间</th><th>业务线</th><th>类别</th><th>假点概率</th><th>判级</th><th>检出</th><th>工单</th><th>耗时</th><th></th>" +
    '</tr></thead><tbody id="recBody"><tr><td colspan="10">加载中…</td></tr></tbody></table></div></div>';
}

function pageAgent() {
  var enabled = state.health && state.health.agent_enabled;
  if (!enabled) {
    return '<div class="card"><h2>缺陷处置 Agent <span class="pill todo">未启用</span></h2>' +
      '<p class="hint">安装依赖并配置 key 后重启服务即可:</p>' +
      '<div class="box info"><code>pip install -r rag_agent/requirements.txt</code><br>' +
      "<code>cp rag_agent/.env.example rag_agent/.env</code> 填入 SILICONFLOW_API_KEY<br>" +
      "<code>python -m rag_agent.build_index</code> 构建 SOP 向量索引</div></div>";
  }
  return '<iframe class="agent" src="/agent/" title="缺陷处置工作台"></iframe>';
}

/* ---------- 交互绑定 ---------- */

function bindDrop(dropId, inputId, onPick) {
  var drop = el(dropId), input = el(inputId);
  if (!drop || !input) return;
  var label = drop.innerHTML;
  function preview(file) {
    onPick(file);
    var url = URL.createObjectURL(file);
    drop.innerHTML = '<img src="' + url + '" alt="预览" /><b>' + esc(file.name) + "</b>";
  }
  drop.onclick = function () { input.click(); };
  input.onchange = function () { if (input.files[0]) preview(input.files[0]); };
  drop.ondragover = function (e) { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = function () { drop.classList.remove("over"); };
  drop.ondrop = function (e) {
    e.preventDefault();
    drop.classList.remove("over");
    if (e.dataTransfer.files[0]) preview(e.dataTransfer.files[0]);
  };
  drop.reset = function () { drop.innerHTML = label; input.value = ""; };
}

function probBars(probs) {
  var entries = Object.keys(probs).map(function (k) { return [k, probs[k]]; });
  entries.sort(function (a, b) { return b[1] - a[1]; });
  return '<div class="bars">' + entries.slice(0, 5).map(function (kv) {
    return '<div class="bar"><label>' + esc(kv[0]) + '</label><span><u style="width:' +
      (kv[1] * 100).toFixed(1) + '%"></u></span><i>' + (kv[1] * 100).toFixed(1) + "%</i></div>";
  }).join("") + "</div>";
}

function bindPcb() {
  bindDrop("dropTest", "filePcbTest", function (f) { pcbFiles.test = f; });
  bindDrop("dropTmpl", "filePcbTmpl", function (f) { pcbFiles.template = f; });

  el("btnPcbClear").onclick = function () {
    pcbFiles = { test: null, template: null };
    el("dropTest").reset(); el("dropTmpl").reset();
    el("pcbMsg").classList.add("hidden");
    el("pcbResult").innerHTML = "";
  };

  el("btnPcbRun").onclick = async function () {
    if (!pcbFiles.test) { showMsg("pcbMsg", "先选待检图"); return; }
    var btn = this;
    btn.disabled = true;
    el("pcbMsg").classList.add("hidden");
    try {
      var fd = new FormData();
      fd.append("image", pcbFiles.test);
      if (pcbFiles.template) fd.append("template", pcbFiles.template);
      fd.append("work_order", el("pcbWo").value);
      fd.append("board_id", el("pcbBoard").value);
      fd.append("avi_item", el("pcbItem").value);
      var r = await getJSON("/pcb/inspect", { method: "POST", body: fd });
      el("pcbResult").innerHTML =
        '<div class="verdict ' + esc(r.status) + '" style="margin-top:12px">' +
        "<div><b>" + esc(r.label) + "</b><div class=\"kv\">" + esc(r.status_text) + "</div></div>" +
        '<div class="kv">假点概率<em>' + (r.ok_prob * 100).toFixed(1) + "%</em></div>" +
        '<div class="kv">类别置信度<em>' + (r.confidence * 100).toFixed(1) + "%</em></div>" +
        '<div class="kv">耗时<em>' + esc(r.elapsed_ms) + " ms</em></div>" +
        '<div class="kv">模板图<em>' + (r.has_template ? "有" : "无") + "</em></div>" +
        '<div class="kv">记录<em>#' + esc(r.id === null ? "-" : r.id) + "</em></div></div>" +
        probBars(r.probs || {});
      refreshStats();
    } catch (e) {
      showMsg("pcbMsg", e.message || e);
    } finally {
      btn.disabled = false;
    }
  };
}

function bindWafer() {
  bindDrop("dropWafer", "fileWafer", function (f) { waferFile = f; });
  el("btnWaferClear").onclick = function () {
    waferFile = null;
    el("dropWafer").reset();
    el("waferMsg").classList.add("hidden");
    el("waferResult").innerHTML = "";
  };
  el("btnWaferRun").onclick = async function () {
    if (!waferFile) { showMsg("waferMsg", "先选硅片图"); return; }
    var btn = this;
    btn.disabled = true;
    el("waferMsg").classList.add("hidden");
    try {
      var fd = new FormData();
      fd.append("image", waferFile);
      fd.append("work_order", el("waferWo").value);
      var r = await getJSON("/wafer/inspect", { method: "POST", body: fd });
      var rows = (r.detections || []).map(function (d) {
        return "<tr><td>" + esc(d.label_text || d.label) + "</td><td>" +
          (d.confidence * 100).toFixed(1) + "%</td><td>" +
          esc((d.bbox_xyxy || []).map(Math.round).join(", ")) + "</td></tr>";
      }).join("");
      el("waferResult").innerHTML =
        '<div class="verdict ' + esc(r.status) + '" style="margin-top:12px">' +
        "<div><b>" + esc(r.num_detections) + ' 处缺陷</b><div class="kv">' +
        (r.num_detections ? "判为 NG" : "未检出,放行") + "</div></div>" +
        '<div class="kv">主缺陷<em>' + esc(r.label || "—") + "</em></div>" +
        '<div class="kv">耗时<em>' + esc(r.elapsed_ms) + " ms</em></div>" +
        '<div class="kv">记录<em>#' + esc(r.id === null ? "-" : r.id) + "</em></div></div>" +
        (rows ? '<table style="margin-top:10px"><thead><tr><th>代码</th><th>置信度</th><th>框 x1,y1,x2,y2</th></tr></thead><tbody>' + rows + "</tbody></table>" : "");
      refreshStats();
    } catch (e) {
      showMsg("waferMsg", e.message || e);
    } finally {
      btn.disabled = false;
    }
  };
}

async function loadRecords() {
  var qs = [];
  var t = el("fTask").value, st = el("fStatus").value, wo = el("fWo").value.trim();
  if (t) qs.push("task=" + encodeURIComponent(t));
  if (st) qs.push("status=" + encodeURIComponent(st));
  if (wo) qs.push("work_order=" + encodeURIComponent(wo));
  qs.push("limit=50");
  el("lnkCsv").href = "/records/export.csv?" + qs.join("&");
  try {
    var rows = await getJSON("/records?" + qs.join("&"));
    if (!rows.length) {
      el("recBody").innerHTML = '<tr><td colspan="10">暂无记录</td></tr>';
      return;
    }
    el("recBody").innerHTML = rows.map(function (r) {
      return "<tr><td>" + esc(r.id) + "</td><td>" + esc((r.created_at || "").replace("T", " ").slice(0, 19)) +
        "</td><td>" + esc(r.task === "pcb" ? "PCB" : "硅片") +
        "</td><td>" + esc(r.label || "—") +
        "</td><td>" + (r.ok_prob === null || r.ok_prob === undefined ? "—" : (r.ok_prob * 100).toFixed(1) + "%") +
        "</td><td>" + statusTag(r.status) +
        "</td><td>" + esc(r.num_detections) +
        "</td><td>" + esc(r.work_order || "—") +
        "</td><td>" + (r.elapsed_ms === null || r.elapsed_ms === undefined ? "—" : Math.round(r.elapsed_ms) + " ms") +
        '</td><td><button class="btn" data-del="' + esc(r.id) + '">删除</button></td></tr>';
    }).join("");
    Array.prototype.forEach.call(document.querySelectorAll("[data-del]"), function (b) {
      b.onclick = async function () {
        try {
          await getJSON("/records/" + b.getAttribute("data-del"), { method: "DELETE" });
          loadRecords(); refreshStats();
        } catch (e) { showMsg("recMsg", e.message || e); }
      };
    });
  } catch (e) {
    showMsg("recMsg", e.message || e);
  }
}

/* ---------- 路由与启动 ---------- */

var PAGES = {
  home: { title: "概览", sub: "两条业务线的接入状态与数据现状", render: pageHome, bind: null },
  pcb: { title: "PCB 假点过滤", sub: "终检 AVI 复判 —— 成对 ROI 分类,把假点自动放行", render: pagePcb, bind: bindPcb },
  wafer: { title: "光伏硅片缺陷检测", sub: "切片后分选 —— 640×640 灰度图目标检测", render: pageWafer, bind: bindWafer },
  records: { title: "检测记录", sub: "两条业务线共用一张记录表,可按业务线/判级/工单筛选", render: pageRecords, bind: bindRecords },
  agent: { title: "缺陷处置", sub: "检出缺陷 → 查 SOP + 查历史 → 多步处置方案,高危动作需人工确认", render: pageAgent, bind: null }
};

function bindRecords() {
  el("btnQuery").onclick = loadRecords;
  el("fWo").onkeydown = function (e) { if (e.key === "Enter") loadRecords(); };
  loadRecords();
}

function renderPage(name) {
  var page = PAGES[name] || PAGES.home;
  state.page = name;
  el("pageTitle").textContent = page.title;
  el("pageSub").textContent = page.sub;
  el("pages").innerHTML = page.render();
  el("kpiRow").classList.toggle("hidden", name === "agent");
  renderKpis();
  Array.prototype.forEach.call(el("nav").querySelectorAll("button"), function (b) {
    b.classList.toggle("active", b.getAttribute("data-page") === name);
  });
  Array.prototype.forEach.call(document.querySelectorAll("[data-goto]"), function (b) {
    b.onclick = function () { renderPage(b.getAttribute("data-goto")); };
  });
  if (page.bind) page.bind();
}

async function refreshStats() {
  try {
    state.stats = await getJSON("/stats");
    renderKpis();
  } catch (_) { /* KPI 拿不到不该挡住主流程 */ }
}

async function boot() {
  try {
    var results = await Promise.all([getJSON("/health"), getJSON("/tasks"), getJSON("/stats")]);
    state.health = results[0];
    state.tasks = results[1];
    state.stats = results[2];
    el("verTag").textContent = state.health.app_en + " v" + state.health.version;
    var pcb = taskOf("pcb"), wafer = taskOf("wafer");
    el("sideFoot").innerHTML = "PCB " + (pcb && pcb.ready ? "✓" : "○") +
      " · 光伏 " + (wafer && wafer.ready ? "✓" : "○") +
      " · Agent " + (state.health.agent_enabled ? "✓" : "○");
  } catch (e) {
    el("pages").innerHTML = '<div class="card"><div class="msg err">' + esc(e.message || e) + "</div></div>";
    return;
  }
  Array.prototype.forEach.call(el("nav").querySelectorAll("button"), function (b) {
    b.onclick = function () { renderPage(b.getAttribute("data-page")); };
  });
  el("btnRefresh").onclick = function () { boot(); };
  renderPage(state.page);
}

boot();
