# -*- coding: utf-8 -*-
"""
风味预测模型
适配: CPU / 低显存 GPU

启动: python gradio_demo.py
"""

import os, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle as pkl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gradio as gr

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 中文字体检测 ──
_CN_FONT = None
for _f in fm.findSystemFonts():
    _fn = os.path.basename(_f).lower()
    if any(k in _fn for k in ["noto", "cjk", "simhei", "simsun", "wqy", "yahei", "droid", "fang"]):
        try:
            _fp = fm.FontProperties(fname=_f)
            if _fp.get_name():
                _CN_FONT = _fp
                break
        except Exception:
            continue
if _CN_FONT is None:
    print("[字体] 未找到中文字体，使用英文标签")
    _SENSORY_LABELS = ["Nut\nFlavor", "     Crispness", "Sweet\nBalance", "Oiliness", "Rancid\nFlavor"]
else:
    print(f"[字体] 使用 {_CN_FONT.get_name()}")
    _SENSORY_LABELS = ["坚果香", "酥脆度", "咸甜平衡", "油腻感", "哈喇味"]

# ── 设备 ──
device = torch.device("cpu")
if torch.cuda.is_available():
    try:
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram >= 4:
            device = torch.device("cuda:0")
    except Exception:
        pass
print(f"[设备] {device}")

# ── 加载 scaler ──
with open(os.path.join(BASE_DIR, "scaler_params.pkl"), "rb") as f:
    sp = pkl.load(f)
scaler_X = sp["scaler_X"]
scaler_y = sp["scaler_y"]
feature_cols = sp["feature_cols"]
label_cols   = sp["label_cols"]

D_IN, D_OUT = 10, 10


class TeacherMLP(nn.Module):
    def __init__(self, d_in=10, d_out=10, hidden=512):
        super().__init__()
        layers = []
        layers += [nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True), nn.Dropout(0.1)]
        for _ in range(6):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True), nn.Dropout(0.1)]
        layers += [nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True), nn.Linear(hidden // 2, d_out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class StudentMLP(nn.Module):
    def __init__(self, d_in=10, d_out=10, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, d_out))
    def forward(self, x): return self.net(x)


@torch.no_grad()
def load_model(ModelClass, path, dev, **kw):
    m = ModelClass(**kw).to("cpu")
    s = torch.load(path, map_location="cpu")
    if all(k.startswith("module.") for k in s.keys()):
        s = {k[7:]: v for k, v in s.items()}
    m.load_state_dict(s, strict=False)
    m.eval()
    return m.to(dev)


print("[加载] 模型 ...")
teacher        = load_model(TeacherMLP, os.path.join(BASE_DIR, "teacher.pth"),           device)
student_distill = load_model(StudentMLP, os.path.join(BASE_DIR, "student_distill.pth"),   device)
student_base   = None
bp = os.path.join(BASE_DIR, "student_baseline.pth")
if os.path.exists(bp):
    student_base = load_model(StudentMLP, bp, device)

SENSORY = ["坚果香", "酥脆度", "咸甜平衡", "油腻感", "哈喇味"]
METRICS = ["油脂氧化货架天数", "每100g原料成本", "总糖", "总脂肪"]
SENSORY_IDX = [label_cols.index(c) for c in SENSORY]
METER_COLORS = {"油脂氧化货架天数": ("#5B86E5", "天"), "每100g原料成本": ("#F39C12", "角"),
                "总糖": ("#27AE60", "g/100g"), "总脂肪": ("#E74C3C", "g/100g")}
METER_ICONS  = {"油脂氧化货架天数": "📅", "每100g原料成本": "💰", "总糖": "🍬", "总脂肪": "🧈"}


@torch.no_grad()
def predict(feat_dict):
    raw = np.array([[feat_dict[c] for c in feature_cols]], dtype=np.float32)
    X = scaler_X.transform(raw)
    Xt = torch.from_numpy(X).to(device)

    def _run(m):
        return scaler_y.inverse_transform(m(Xt).cpu().numpy()).flatten()

    return _run(teacher), _run(student_distill), (_run(student_base) if student_base else None), raw[0]


# ── Radar ──
def make_radar(tv, dv, bv=None):
    N = len(_SENSORY_LABELS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw=dict(polar=True),
                           facecolor="#FEF9FF")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    xtick_labels = _SENSORY_LABELS
    ax.set_xticklabels(xtick_labels, fontsize=10, fontweight="bold", color="#333",
                       fontproperties=_CN_FONT)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color="#666")
    title = "Sensory Score Radar" if _CN_FONT is None else "感官评分雷达图"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20, color="#333",
                 fontproperties=_CN_FONT)
    ax.grid(True, alpha=0.25, color="#BBB")

    data = [
        ("本体 8层" if _CN_FONT else "Teacher", tv, "#E74C3C", "-o", 0.10),
        ("蒸馏 2层" if _CN_FONT else "Distill", dv, "#2E86C1", "-s", 0.12),
    ]
    if bv is not None:
        b_label = "基线 2层" if _CN_FONT else "Baseline"
        data.append((b_label, bv, "#95A5A6", ":^", 0.06))
    for label, vals, color, marker, alpha in data:
        v = vals[SENSORY_IDX].tolist() + [vals[SENSORY_IDX][0]]
        ax.plot(angles, v, marker, color=color, linewidth=2.2, label=label, markersize=6)
        ax.fill(angles, v, alpha=alpha, color=color)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=9,
              frameon=True, facecolor="white", edgecolor="#DDD", prop=_CN_FONT)
    fig.tight_layout()
    return fig


# ── 核心回调 ──
def on_change(*vals):
    fd = {k: v for k, v in zip(feature_cols, vals)}
    t, d, b, raw = predict(fd)
    radar = make_radar(t, d, b)

    # 卡片
    def make_cards(tvec, dvec):
        parts = ""
        for ci, c in enumerate(METRICS):
            i = label_cols.index(c)
            color, unit = METER_COLORS[c]
            icon = METER_ICONS[c]
            tv = tvec[i]
            dv = dvec[i]
            if "天数" in c:
                ts, ds = f"{tv:.0f}", f"{dv:.0f}"
            elif "成本" in c or "脂肪" in c:
                ts, ds = f"{tv:.1f}", f"{dv:.1f}"
            else:
                ts, ds = f"{tv:.2f}", f"{dv:.2f}"
            parts += f"""
<div style="background:#FFFDFF;border-radius:12px;padding:12px 10px;box-shadow:0 2px 8px rgba(0,0,0,0.04);text-align:center;min-width:110px;flex:1;border:1px solid #F0E6F6;">
<div style="font-size:18px;">{icon}</div>
<div style="font-size:11px;color:#555;margin:2px 0;font-weight:500;">{c}</div>
<div style="display:flex;justify-content:center;gap:14px;margin-top:4px;">
<div><span style="font-size:10px;color:#888;">教师</span><br><span style="font-size:17px;font-weight:700;color:{color};">{ts}</span></div>
<div style="color:#DDD;">|</div>
<div><span style="font-size:10px;color:#888;">蒸馏</span><br><span style="font-size:17px;font-weight:700;color:{color};">{ds}</span></div>
</div>
<div style="font-size:10px;color:#AAA;">{unit}</div>
</div>"""
        return f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{parts}</div>'

    cards_html = make_cards(t, d)

    # 感官评分表
    def make_sensory(tv, dv):
        rows = ""
        for sc in SENSORY:
            i = label_cols.index(sc)
            st = "★" * max(1, min(10, int(round(tv[i]))))
            sd = "★" * max(1, min(10, int(round(dv[i]))))
            rows += f"<tr><td style='padding:6px 10px;font-weight:500;color:#333;'>{sc}</td>"
            rows += f"<td style='padding:6px 10px;font-weight:600;color:#C0392B;text-align:center;'>{tv[i]:.1f}</td>"
            rows += f"<td style='padding:6px 8px;font-size:13px;color:#C0392B;'>{st}</td>"
            rows += f"<td style='padding:6px 10px;font-weight:600;color:#1A6FA0;text-align:center;'>{dv[i]:.1f}</td>"
            rows += f"<td style='padding:6px 8px;font-size:13px;color:#1A6FA0;'>{sd}</td></tr>"
        return f"""<table style="width:100%;border-collapse:collapse;font-size:14px;color:#333;">
<tr style="background:#8E44AD;color:white;">
<th style="padding:8px 12px;text-align:left;font-weight:500;">感官指标</th>
<th style="padding:8px 12px;text-align:center;font-weight:500;" colspan="2">本体 8层</th>
<th style="padding:8px 12px;text-align:center;font-weight:500;" colspan="2">蒸馏 2层</th>
</tr>{rows}</table>"""

    sensory_html = make_sensory(t, d)

    # 模型 info
    teacher_box = """<div style="background:linear-gradient(135deg,#FDF2F8,#FFF5F7);border-radius:14px;padding:16px;border:1px solid #FBCFE8;box-shadow:0 2px 10px rgba(0,0,0,0.04);">
<div style="font-size:14px;color:#BE185D;font-weight:700;margin-bottom:6px;">🏫 本体模型</div>
<div style="font-size:13px;color:#333;">8 层深层 MLP  ·  隐藏层 512</div>
<div style="font-size:12px;color:#666;margin-top:3px;">参数量 ~3.7M  ·  高精度预测</div>
</div>"""
    distill_box = """<div style="background:linear-gradient(135deg,#EDF4FF,#F5F9FF);border-radius:14px;padding:16px;border:1px solid #BFDBFE;box-shadow:0 2px 10px rgba(0,0,0,0.04);">
<div style="font-size:14px;color:#1D4ED8;font-weight:700;margin-bottom:6px;">⚡ 蒸馏模型</div>
<div style="font-size:13px;color:#333;">2 层浅层 MLP  ·  隐藏层 64</div>
<div style="font-size:12px;color:#666;margin-top:3px;">参数量 ~5.4K  ·  轻量部署</div>
</div>"""

    return radar, sensory_html, cards_html, teacher_box, distill_box


# ── 导出 ──
def export_csv(*vals):
    fd = {k: v for k, v in zip(feature_cols, vals)}
    t, d, b, raw = predict(fd)
    rows = []
    for i, c in enumerate(feature_cols):
        rows.append({"类别": "配方输入", "指标": c, "数值": f"{raw[i]:.4f}"})
    for i, c in enumerate(label_cols):
        rows.append({"类别": "本体预测", "指标": c, "数值": f"{t[i]:.4f}"})
        rows.append({"类别": "蒸馏预测", "指标": c, "数值": f"{d[i]:.4f}"})
    df = pd.DataFrame(rows)
    path = os.path.join(BASE_DIR, "配方研发报告.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ── Slider 默认值 ──
SLIDER_CFG = {
    "葵花籽占比": (0.35, 0.70, 0.01, 0.52),
    "巴旦木占比": (0.05, 0.25, 0.01, 0.15),
    "核桃占比":   (0.05, 0.20, 0.01, 0.12),
    "白砂糖":     (0.010, 0.080, 0.001, 0.044),
    "代糖":       (0.002, 0.015, 0.001, 0.009),
    "食用盐":     (0.003, 0.012, 0.001, 0.007),
    "复合香料":   (0.002, 0.010, 0.001, 0.006),
    "烘烤温度":   (120, 180, 1, 150),
    "烘烤时长":   (10.0, 35.0, 0.5, 22.0),
    "冷却时间":   (20, 120, 1, 60),
}

# ── CSS ──
CSS = """
.gradio-container {max-width:1500px!important; margin:0 auto; background:linear-gradient(135deg,#FDF2F8,#F3E8FF,#EDF4FF)!important;}
.app-title {text-align:center; background:linear-gradient(135deg,#7C3AED,#DB2777); color:white;
            padding:24px 32px; border-radius:16px; margin:0 0 20px 0; box-shadow:0 4px 20px rgba(0,0,0,0.12);}
.app-title h1 {margin:0; font-size:26px; font-weight:700; color:white;}
.app-title p {margin:6px 0 0 0; font-size:14px; opacity:0.9; color:white;}
label, .label-text {color:#333!important; font-weight:500!important;}
.section-card {background:rgba(255,255,255,0.85); backdrop-filter:blur(8px); border-radius:16px; padding:18px;
               box-shadow:0 2px 12px rgba(0,0,0,0.05); margin-bottom:14px; border:1px solid rgba(255,255,255,0.6);}
.section-title {font-size:15px; font-weight:700; color:#333; margin-bottom:12px;
                border-bottom:2px solid #A855F7; padding-bottom:6px; display:flex; align-items:center; gap:8px;}
footer {display:none!important;}
"""

# ═══════════════════════════════════════════════════════════════
#  Gradio UI
# ═══════════════════════════════════════════════════════════════
with gr.Blocks(css=CSS, title="风味预测模型") as demo:
    gr.HTML("""
    <div class="app-title">
        <h1>🥜 风味预测模型</h1>
        <p>原料配比 · 烘烤工艺 → 感官评分 · 货架预测 · 成本分析 &nbsp;|&nbsp; 教师模型 (8层) vs 蒸馏学生 (2层)</p>
    </div>""")

    # ── 滑动条 ──
    sliders = {}
    with gr.Column():
        gr.HTML('<div class="section-card"><div class="section-title">📋 配方参数调节</div>')
        with gr.Row():
            with gr.Column(scale=1):
                for c in feature_cols[:5]:
                    lo, hi, st, dft = SLIDER_CFG[c]
                    sliders[c] = gr.Slider(minimum=lo, maximum=hi, step=st, value=dft, label=c)
            with gr.Column(scale=1):
                for c in feature_cols[5:]:
                    lo, hi, st, dft = SLIDER_CFG[c]
                    sliders[c] = gr.Slider(minimum=lo, maximum=hi, step=st, value=dft, label=c)
        gr.HTML('</div>')

    # ── 雷达 + 模型 info ──
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            with gr.Group():
                radar = gr.Plot(label=None, show_label=False)
        with gr.Column(scale=2):
            gr.HTML('<div style="display:flex;flex-direction:column;gap:10px;height:100%;justify-content:center;">')
            teacher_info = gr.HTML()
            distill_info = gr.HTML()
            gr.HTML('</div>')

    # ── 感官评分 ──
    with gr.Column():
        gr.HTML('<div class="section-card"><div class="section-title">⭐ 感官评分对比</div>')
        sensory_table = gr.HTML()
        gr.HTML('</div>')

    # ── 理化指标 ──
    with gr.Column():
        gr.HTML('<div class="section-card"><div class="section-title">📊 理化 & 成本指标对比</div>')
        metric_cards = gr.HTML()
        gr.HTML('</div>')

    # ── 导出 ──
    with gr.Row():
        export_btn = gr.Button("📥 导出 CSV 研发报告", variant="primary", size="lg", scale=2)
        export_file = gr.File(label="下载", scale=3)

    # ── 绑定 ──
    inputs = [sliders[c] for c in feature_cols]
    outputs = [radar, sensory_table, metric_cards, teacher_info, distill_info]

    for s in inputs:
        s.change(fn=on_change, inputs=inputs, outputs=outputs)
    export_btn.click(fn=export_csv, inputs=inputs, outputs=export_file)
    demo.load(fn=on_change, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)