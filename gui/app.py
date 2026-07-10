"""
gui/app.py  — CRC-AI Clinical GUI  (v3 — reorganised layout)
─────────────────────────────────────────────────────────────────────────────
Changes from v2:
  • Verdict/ANALYSING banner REMOVED from centre top — images now fill that space
  • 2×2 image grid is taller (fills full centre area)
  • Confidence gauge is taller / wider so the % text never overlaps the arc
  • Classified class + verdict displayed BELOW the gauge in the right pane
  • Uncertainty row stays under class display
  • Right pane re-ordered: gauge → class/verdict card → uncertainty → prob bars → metadata
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import io
import base64
import threading
import math
import json
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ─── Design tokens ────────────────────────────────────────────────────────

BG_ROOT    = "#0B0F14"
BG_TOPBAR  = "#080C10"
BG_PANEL   = "#111620"
BG_CARD    = "#171D27"
BG_CARD2   = "#1C2333"
BORDER_CLR = "#2A3447"

ACCENT     = "#4D9EFF"
MAL_CLR    = "#FF4D4D"
BEN_CLR    = "#2ECC71"
WARN_CLR   = "#F0A500"

TEXT1      = "#E8EFF8"
TEXT2      = "#7E8FA8"
TEXT3      = "#3D4F68"

FONT_VERDICT = ("Segoe UI Black",  22)
FONT_CLASS   = ("Segoe UI Semibold", 15)
FONT_CONF    = ("Segoe UI Black",  34)
FONT_H2      = ("Segoe UI Semibold", 11)
FONT_BODY    = ("Segoe UI", 10)
FONT_SMALL   = ("Segoe UI", 9)
FONT_MONO    = ("Consolas", 9)
FONT_CAP     = ("Segoe UI Semibold", 8)

CLASSES_LIST = [
    "Adenocarcinoma", "Serrated Adenoma", "Polyp",
    "Benign", "Low Grade IN", "High Grade IN",
]
CLASS_COLORS = {
    "Adenocarcinoma":   "#FF4D4D",
    "Serrated Adenoma": "#FF8C42",
    "Polyp":            "#F0C040",
    "Benign":           "#2ECC71",
    "Low Grade IN":     "#4D9EFF",
    "High Grade IN":    "#B07FFF",
}
MODEL_NAMES = [
    "SwinTransformer", "VisionTransformer", "ConvNeXt",
    "DenseNet121", "DenseNet201", "EfficientNetB4",
    "EfficientNetV2", "InceptionResNetV2", "Xception",
    "ResNet50", "ResNet101", "SEResNet", "CBAMResNet",
    "BEiT", "DeiT",
]

# ─── Helpers ──────────────────────────────────────────────────────────────

def _b64_to_pil(b64_str: str, size: tuple) -> Optional[ImageTk.PhotoImage]:
    if not b64_str:
        return None
    try:
        buf = base64.b64decode(b64_str)
        pil = Image.open(io.BytesIO(buf)).convert("RGB").resize(size, Image.LANCZOS)
        return ImageTk.PhotoImage(pil)
    except Exception:
        return None


def _placeholder(w: int, h: int, label: str = "") -> ImageTk.PhotoImage:
    img = Image.new("RGB", (w, h), BG_CARD)
    d   = ImageDraw.Draw(img)
    dash = 6
    for x in range(0, w, dash * 2):
        d.line([(x, 1), (min(x + dash, w), 1)], fill=BORDER_CLR, width=1)
        d.line([(x, h - 2), (min(x + dash, w), h - 2)], fill=BORDER_CLR, width=1)
    for y in range(0, h, dash * 2):
        d.line([(1, y), (1, min(y + dash, h))], fill=BORDER_CLR, width=1)
        d.line([(w - 2, y), (w - 2, min(y + dash, h))], fill=BORDER_CLR, width=1)
    if label:
        d.text((w // 2, h // 2), label, fill=TEXT3, anchor="mm")
    return ImageTk.PhotoImage(img)


def _draw_linear_meter(canvas: tk.Canvas, value: float,
                       color: str, n_tta: int = 8) -> None:
    """
    Draw a horizontal linear confidence meter.

    Layout (canvas 408 × 145):
      y=24   tick labels  0  10  20 … 100
      y=34   tick lines
      y=38   track bar top   (bar height 18px)
      y=56   track bar bottom
      y=56   marker triangle tip
      y=90   large % readout  (FONT_CONF 34pt ≈ 30px tall → bottom ~105)
      y=125  TTA label        (9pt ≈ 12px → bottom ~137, fits in 145)
    """
    canvas.delete("gauge")
    W      = int(canvas["width"])
    PAD_L  = 18
    PAD_R  = 18
    BAR_X0 = PAD_L
    BAR_X1 = W - PAD_R
    BAR_W  = BAR_X1 - BAR_X0
    BAR_Y0 = 38          # bar top
    BAR_Y1 = 56          # bar bottom  (18px tall)
    R      = 6           # corner radius
    FILL_X = BAR_X0 + int(value * BAR_W)

    # ── Colour zones (green → amber → red gradient ticks) ────────────
    ZONES = [
        (0.00, 0.50, "#2ECC71"),
        (0.50, 0.75, "#F0A500"),
        (0.75, 1.00, "#FF4D4D"),
    ]

    # ── Tick labels ───────────────────────────────────────────────────
    for t in range(0, 101, 10):
        tx = BAR_X0 + int(t / 100 * BAR_W)
        canvas.create_line(tx, BAR_Y0 - 5, tx, BAR_Y0 - 1,
                           fill=TEXT3, width=1, tags="gauge")
        canvas.create_text(tx, BAR_Y0 - 9, text=str(t),
                           font=("Consolas", 7), fill=TEXT3,
                           anchor="s", tags="gauge")

    # ── Track background (rounded rect via overlapping rects+ovals) ──
    canvas.create_rectangle(BAR_X0 + R, BAR_Y0, BAR_X1 - R, BAR_Y1,
                             fill=BG_CARD2, outline="", tags="gauge")
    canvas.create_rectangle(BAR_X0, BAR_Y0 + R, BAR_X1, BAR_Y1 - R,
                             fill=BG_CARD2, outline="", tags="gauge")
    for ox, oy in [(BAR_X0, BAR_Y0), (BAR_X1 - 2*R, BAR_Y0),
                   (BAR_X0, BAR_Y1 - 2*R), (BAR_X1 - 2*R, BAR_Y1 - 2*R)]:
        canvas.create_oval(ox, oy, ox + 2*R, oy + 2*R,
                           fill=BG_CARD2, outline="", tags="gauge")

    # Track border
    canvas.create_rectangle(BAR_X0, BAR_Y0, BAR_X1, BAR_Y1,
                             fill="", outline=BORDER_CLR, width=1, tags="gauge")

    # ── Zone-coloured fill ────────────────────────────────────────────
    if value > 0.001:
        prev_x = BAR_X0
        for z0, z1, zcol in ZONES:
            seg_x0 = BAR_X0 + int(z0 * BAR_W)
            seg_x1 = BAR_X0 + int(z1 * BAR_W)
            draw_x1 = min(FILL_X, seg_x1)
            if draw_x1 > prev_x:
                canvas.create_rectangle(prev_x, BAR_Y0 + 1,
                                        draw_x1, BAR_Y1 - 1,
                                        fill=zcol, outline="", tags="gauge")
            prev_x = seg_x1
            if FILL_X <= seg_x1:
                break

    # ── Marker: vertical line + downward triangle pointer ─────────────
    if value > 0.001:
        MX = FILL_X
        canvas.create_line(MX, BAR_Y0 - 1, MX, BAR_Y1 + 1,
                           fill=color, width=2, tags="gauge")
        # Triangle above bar pointing down
        tip_y = BAR_Y0 - 1
        canvas.create_polygon(MX - 6, tip_y - 9,
                               MX + 6, tip_y - 9,
                               MX, tip_y,
                               fill=color, outline="", tags="gauge")

    # ── Large percentage value — 20px gap below bar ───────────────────
    canvas.create_text(W // 2, BAR_Y1 + 34,
                       text=f"{value * 100:.1f}%",
                       font=FONT_CONF, fill=color, tags="gauge")

    # ── TTA label — 14px gap below % text ────────────────────────────
    canvas.create_text(W // 2, BAR_Y1 + 70,
                       text=f"{n_tta}-view TTA",
                       font=FONT_SMALL, fill=TEXT2, tags="gauge")


# ─── Main Application ─────────────────────────────────────────────────────

class ClinicalGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("CRC-AI  |  Colorectal Cancer Diagnostic System")
        self.configure(bg=BG_ROOT)
        self.geometry("1760x980")
        self.minsize(1400, 820)
        self.resizable(True, True)

        self._img_path       = None
        self._result         = None
        self._photos         = {}
        self._history        = []
        self._spinning       = False
        self._spin_angle     = 0
        self._pipeline_ready = False

        self._build_ui()
        self._tick_spinner()
        self.after(200, self._try_import_pipeline)

    # ── Pipeline ─────────────────────────────────────────────────────

    def _try_import_pipeline(self):
        try:
            from inference.pipeline import run_inference
            self._run_inference = run_inference
            self._pipeline_ready = True
            self._status("✓  Pipeline loaded — open an image to begin.")
        except ImportError as e:
            self._run_inference  = self._demo_inference
            self._pipeline_ready = True
            self._status(f"⚠  Demo mode: {e}")

    def _demo_inference(self, image_input, model_name="ResNet50",
                        n_tta=8, temperature=1.5, cam_method="Ensemble",
                        device=None):
        import random, time as T
        T.sleep(1.2)
        probs = {c: random.random() for c in CLASSES_LIST}
        s     = sum(probs.values())
        probs = {k: v / s for k, v in probs.items()}
        pred  = max(probs, key=probs.get)

        class _R:
            predicted_class  = pred
            predicted_idx    = CLASSES_LIST.index(pred)
            is_malignant     = pred != "Benign"
            verdict          = "MALIGNANT" if pred != "Benign" else "BENIGN"
            confidence       = probs[pred]
            uncertainty      = 0.03
            staging          = "[DEMO]"
            probabilities    = probs
            per_view_probs   = []
            n_tta_views      = n_tta
            model_name       = model_name
            inference_time_s = 1.2
            image_shape      = (224, 224, 3)
            original_b64 = heatmap_b64 = overlay_b64 = panel_b64 = ""
            cam_maps = {}
        return _R()

    # ── Top-level layout ──────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────
        top = tk.Frame(self, bg=BG_TOPBAR, height=52)
        top.pack(side="top", fill="x")
        top.pack_propagate(False)

        tk.Label(top, text="CRC", font=("Segoe UI Black", 17),
                 fg=ACCENT, bg=BG_TOPBAR).pack(side="left", padx=(18, 2))
        tk.Label(top, text="-AI", font=("Segoe UI Light", 17),
                 fg=TEXT1, bg=BG_TOPBAR).pack(side="left")
        tk.Label(top,
                 text="  Colorectal Cancer Histopathology Diagnostic System",
                 font=("Segoe UI", 10), fg=TEXT2,
                 bg=BG_TOPBAR).pack(side="left", padx=6)

        self._status_var = tk.StringVar(value="Initialising…")
        tk.Label(top, textvariable=self._status_var,
                 font=FONT_SMALL, fg=TEXT3,
                 bg=BG_TOPBAR).pack(side="right", padx=18)

        # ── Body: three columns ───────────────────────────────────────
        body = tk.Frame(self, bg=BG_ROOT)
        body.pack(fill="both", expand=True, padx=10, pady=(8, 8))

        left = tk.Frame(body, bg=BG_ROOT, width=340)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG_ROOT, width=410)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        centre = tk.Frame(body, bg=BG_ROOT)
        centre.pack(side="left", fill="both", expand=True, padx=10)

        self._build_left(left)
        self._build_centre(centre)
        self._build_right(right)

    # ── LEFT ──────────────────────────────────────────────────────────

    def _build_left(self, p):
        self._sec(p, "IMAGE INPUT")
        inp = self._cframe(p)
        inp.pack(fill="x", pady=(0, 10))

        clk = tk.Label(inp,
                       text="📂   Click to open image\n\nPNG  ·  JPG  ·  TIFF  ·  BMP",
                       font=("Segoe UI", 9), fg=TEXT2, bg=BG_CARD,
                       cursor="hand2", justify="center", pady=16)
        clk.pack(fill="x")
        clk.bind("<Button-1>", lambda e: self._browse())

        self._thumb_lbl = tk.Label(inp, bg=BG_CARD)
        self._thumb_lbl.pack(pady=(4, 0))

        self._fname_var = tk.StringVar(value="No image loaded")
        tk.Label(inp, textvariable=self._fname_var, font=FONT_SMALL,
                 fg=TEXT2, bg=BG_CARD, wraplength=300).pack(pady=(2, 8))

        self._sec(p, "MODEL SETTINGS")
        cfg = self._cframe(p)
        cfg.pack(fill="x", pady=(0, 10))
        self._combo(cfg, "Backbone",    MODEL_NAMES,      "CBAMResNet", "_model_var")
        self._combo(cfg, "TTA Views",   ["1", "4", "8"],  "8",          "_tta_var")
        self._combo(cfg, "CAM Method",  ["GradCAM", "GradCAM++", "EigenCAM",
                                          "LayerCAM", "Ensemble"],
                    "Ensemble", "_cam_var")
        self._combo(cfg, "Temperature", ["1.0", "1.5", "2.0"], "1.5",   "_temp_var")

        btn_card = self._cframe(p)
        btn_card.pack(fill="x", pady=(0, 10))
        self._run_btn = tk.Button(
            btn_card, text="▶   RUN ANALYSIS",
            font=("Segoe UI Semibold", 11),
            bg=ACCENT, fg=BG_ROOT, activebackground="#79BAFF",
            relief="flat", cursor="hand2", pady=12,
            command=self._run)
        self._run_btn.pack(fill="x")
        self._exp_btn = tk.Button(
            btn_card, text="⬇   Export Report",
            font=FONT_SMALL, bg=BG_CARD2, fg=TEXT2,
            activebackground=BORDER_CLR, relief="flat",
            cursor="hand2", pady=7, command=self._export)
        self._exp_btn.pack(fill="x", pady=(4, 0))

        spin_row = tk.Frame(p, bg=BG_ROOT)
        spin_row.pack(fill="x", pady=4)
        self._spin_cv = tk.Canvas(spin_row, width=36, height=36,
                                   bg=BG_ROOT, highlightthickness=0)
        self._spin_cv.pack(side="left", padx=4)

        self._sec(p, "SESSION HISTORY")
        hist = self._cframe(p)
        hist.pack(fill="both", expand=True, pady=(0, 4))
        self._hist_box = tk.Listbox(
            hist, bg=BG_CARD, fg=TEXT2,
            selectbackground=ACCENT, selectforeground=BG_ROOT,
            font=FONT_SMALL, relief="flat", bd=0,
            highlightthickness=0, activestyle="none")
        self._hist_box.pack(fill="both", expand=True, padx=6, pady=6)
        self._hist_box.bind("<<ListboxSelect>>", self._on_hist)

    # ── CENTRE — image grid only, no banner ───────────────────────────

    def _build_centre(self, p):
        """
        The centre panel is ONLY the 2×2 image grid.
        The verdict banner has been removed so images get full height.
        """
        grid = tk.Frame(p, bg=BG_ROOT)
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=1, uniform="c")
        grid.columnconfigure(1, weight=1, uniform="c")
        grid.rowconfigure(0, weight=1, uniform="r")
        grid.rowconfigure(1, weight=1, uniform="r")

        panels = [
            ("Original",  "_img_orig", 0, 0),
            ("Heatmap",   "_img_heat", 0, 1),
            ("Overlay",   "_img_over", 1, 0),
            ("TTA Views", "_img_tta",  1, 1),
        ]
        self._img_labels = {}

        for title, key, row, col in panels:
            outer = tk.Frame(grid, bg=BG_CARD,
                             highlightbackground=BORDER_CLR,
                             highlightthickness=1)
            outer.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

            hdr = tk.Frame(outer, bg=BG_CARD2, height=26)
            hdr.pack(fill="x")
            hdr.pack_propagate(False)
            tk.Label(hdr, text=title, font=FONT_CAP,
                     fg=TEXT2, bg=BG_CARD2).pack(side="left", padx=10, pady=4)

            lbl = tk.Label(outer, bg=BG_CARD, cursor="crosshair")
            lbl.pack(fill="both", expand=True, padx=4, pady=4)
            self._img_labels[key] = lbl
            lbl.bind("<Configure>", lambda e, k=key: self._resize_panel(k, e))

        self._init_placeholders()

    # ── RIGHT — gauge (tall) → class card → uncertainty → probs → meta ─

    def _build_right(self, p):

        # ── 1. Confidence gauge ───────────────────────────────────────
        self._sec(p, "CONFIDENCE")
        gc = self._cframe(p)
        gc.pack(fill="x", pady=(0, 6))

        # Linear meter: 408 wide, 145 tall
        GAUGE_W, GAUGE_H = 408, 145

        self._gauge_cv = tk.Canvas(gc, width=GAUGE_W, height=GAUGE_H,
                                    bg=BG_CARD, highlightthickness=0)
        self._gauge_cv.pack(padx=0, pady=(8, 6))
        _draw_linear_meter(self._gauge_cv, 0.0, TEXT3, 8)

        # ── 2. Classified class + verdict card ────────────────────────
        self._sec(p, "CLASSIFICATION RESULT")
        rc = self._cframe(p)
        rc.pack(fill="x", pady=(0, 6))

        result_inner = tk.Frame(rc, bg=BG_CARD)
        result_inner.pack(fill="x", padx=12, pady=10)

        # Verdict row
        verdict_row = tk.Frame(result_inner, bg=BG_CARD)
        verdict_row.pack(fill="x", pady=(0, 4))
        tk.Label(verdict_row, text="VERDICT:", font=FONT_CAP,
                 fg=TEXT3, bg=BG_CARD, anchor="w").pack(side="left")
        self._verdict_var = tk.StringVar(value="AWAITING ANALYSIS")
        self._verdict_lbl = tk.Label(verdict_row, textvariable=self._verdict_var,
                                      font=FONT_VERDICT, fg=TEXT3, bg=BG_CARD)
        self._verdict_lbl.pack(side="right")

        # Class row
        class_row = tk.Frame(result_inner, bg=BG_CARD)
        class_row.pack(fill="x", pady=(0, 4))
        tk.Label(class_row, text="CLASS:", font=FONT_CAP,
                 fg=TEXT3, bg=BG_CARD, anchor="w").pack(side="left")
        self._class_var = tk.StringVar(value="—")
        self._class_lbl = tk.Label(class_row, textvariable=self._class_var,
                                    font=FONT_CLASS, fg=TEXT2, bg=BG_CARD)
        self._class_lbl.pack(side="right")

        # Uncertainty row
        unc_row = tk.Frame(result_inner, bg=BG_CARD)
        unc_row.pack(fill="x")
        tk.Label(unc_row, text="Uncertainty (±):", font=FONT_SMALL,
                 fg=TEXT2, bg=BG_CARD).pack(side="left")
        self._unc_var = tk.StringVar(value="—")
        tk.Label(unc_row, textvariable=self._unc_var,
                 font=FONT_MONO, fg=WARN_CLR, bg=BG_CARD).pack(side="right")

        # ── 3. Class probabilities ────────────────────────────────────
        self._sec(p, "CLASS PROBABILITIES")
        pc = self._cframe(p)
        pc.pack(fill="x", pady=(0, 10))

        self._prob_cv = tk.Canvas(pc, bg=BG_CARD,
                                   height=218, width=408,
                                   highlightthickness=0)
        self._prob_cv.pack(fill="x", padx=0, pady=6)
        self._draw_probs({c: 0.0 for c in CLASSES_LIST})

        # ── 4. Analysis metadata ──────────────────────────────────────
        self._sec(p, "ANALYSIS METADATA")
        mc = self._cframe(p)
        mc.pack(fill="x", pady=(0, 10))

        self._meta = {}
        for key in ["Model", "TTA Views", "CAM Method",
                    "Inference Time", "Image Shape"]:
            row = tk.Frame(mc, bg=BG_CARD)
            row.pack(fill="x", padx=12, pady=3)
            tk.Label(row, text=key + ":", font=FONT_SMALL,
                     fg=TEXT3, bg=BG_CARD, width=15,
                     anchor="w").pack(side="left")
            var = tk.StringVar(value="—")
            tk.Label(row, textvariable=var, font=FONT_MONO,
                     fg=TEXT2, bg=BG_CARD).pack(side="left")
            self._meta[key] = var

    # ── Widget helpers ─────────────────────────────────────────────────

    def _sec(self, parent, text: str):
        tk.Label(parent, text=text, font=FONT_CAP, fg=TEXT3,
                 bg=BG_ROOT, anchor="w", padx=2, pady=3).pack(fill="x")

    def _cframe(self, parent) -> tk.Frame:
        return tk.Frame(parent, bg=BG_CARD,
                        highlightbackground=BORDER_CLR, highlightthickness=1)

    def _combo(self, parent, label: str, options: list,
               default: str, attr: str):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", padx=10, pady=4)
        tk.Label(row, text=label + ":", font=FONT_SMALL,
                 fg=TEXT2, bg=BG_CARD, width=13, anchor="w").pack(side="left")
        var = tk.StringVar(value=default)
        cb  = ttk.Combobox(row, textvariable=var, values=options,
                            state="readonly", font=FONT_SMALL, width=16)
        cb.pack(side="right")
        setattr(self, attr, var)

    # ── Probability bars ───────────────────────────────────────────────

    def _draw_probs(self, probs: dict, highlight: str = ""):
        cv    = self._prob_cv
        cv.delete("all")
        W     = 406
        bar_h = 20
        gap   = 15
        x_lbl = 10
        x_bar = 150
        bar_w = W - x_bar - 48

        for i, cls in enumerate(CLASSES_LIST):
            y   = i * (bar_h + gap) + 8
            p   = probs.get(cls, 0.0)
            col = CLASS_COLORS.get(cls, ACCENT)
            hl  = cls == highlight

            if hl:
                cv.create_rectangle(0, y - 3, W, y + bar_h + 3,
                                     fill="#182030", outline="")
            cv.create_text(x_lbl, y + bar_h // 2, text=cls,
                            anchor="w", font=FONT_SMALL,
                            fill=TEXT1 if hl else TEXT2)
            cv.create_rectangle(x_bar, y, x_bar + bar_w, y + bar_h,
                                  fill=BORDER_CLR, outline="")
            filled = int(p * bar_w)
            if filled > 1:
                cv.create_rectangle(x_bar, y, x_bar + filled, y + bar_h,
                                     fill=col, outline="")
            cv.create_text(x_bar + bar_w + 6, y + bar_h // 2,
                            text=f"{p * 100:.1f}%", anchor="w",
                            font=FONT_MONO,
                            fill=TEXT1 if hl else TEXT2)

    # ── Spinner ────────────────────────────────────────────────────────

    def _tick_spinner(self):
        if self._spinning:
            c  = self._spin_cv
            c.delete("all")
            cx, cy, r = 18, 18, 12
            for i in range(8):
                ang = math.radians(self._spin_angle + i * 45)
                x   = cx + r * math.cos(ang)
                y   = cy + r * math.sin(ang)
                gv  = int(55 + (i + 1) / 8 * 160)
                c.create_oval(x - 3, y - 3, x + 3, y + 3,
                               fill=f"#{gv:02x}{gv:02x}{gv:02x}",
                               outline="")
            self._spin_angle = (self._spin_angle + 15) % 360
        self.after(50, self._tick_spinner)

    # ── Panel resize ───────────────────────────────────────────────────

    def _resize_panel(self, key: str, event):
        if not self._result:
            return
        b64_map = {
            "_img_orig": getattr(self._result, "original_b64", ""),
            "_img_heat": getattr(self._result, "heatmap_b64",  ""),
            "_img_over": getattr(self._result, "overlay_b64",  ""),
            "_img_tta":  getattr(self._result, "panel_b64",    ""),
        }
        b64 = b64_map.get(key, "")
        if not b64:
            return
        w  = max(event.width  - 8, 60)
        h  = max(event.height - 8, 60)
        ph = _b64_to_pil(b64, (w, h))
        if ph:
            self._img_labels[key].config(image=ph)
            self._photos[key] = ph

    def _init_placeholders(self):
        labels = {"_img_orig": "Original", "_img_heat": "Heatmap",
                  "_img_over": "Overlay",  "_img_tta":  "TTA Views"}
        for key, txt in labels.items():
            ph = _placeholder(320, 300, txt)
            self._img_labels[key].config(image=ph)
            self._photos[f"ph_{key}"] = ph

    # ── Image loading ──────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Histopathology Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                       ("All files", "*.*")])
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        self._img_path = path
        fname = Path(path).name
        self._fname_var.set(fname[:42] + "…" if len(fname) > 42 else fname)

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((300, 130), Image.LANCZOS)
            ph  = ImageTk.PhotoImage(img)
            self._thumb_lbl.config(image=ph)
            self._photos["thumb"] = ph
        except Exception:
            pass

        try:
            bgr = cv2.imread(path)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                lbl = self._img_labels["_img_orig"]
                w   = max(lbl.winfo_width(),  320)
                h   = max(lbl.winfo_height(), 300)
                pil = Image.fromarray(rgb).resize((w, h), Image.LANCZOS)
                ph  = ImageTk.PhotoImage(pil)
                lbl.config(image=ph)
                self._photos["orig_load"] = ph
        except Exception:
            pass

        self._status(f"Loaded: {fname}  — press RUN ANALYSIS")

    # ── Analysis ───────────────────────────────────────────────────────

    def _run(self):
        if not self._img_path:
            messagebox.showwarning("No Image", "Please open an image first.")
            return
        if not self._pipeline_ready:
            messagebox.showinfo("Loading", "Pipeline still loading, please wait.")
            return
        self._run_btn.config(state="disabled", text="Analysing…")
        self._spinning = True
        self._status("Running inference…")
        threading.Thread(target=self._infer_thread, daemon=True).start()

    def _infer_thread(self):
        try:
            result = self._run_inference(
                self._img_path,
                model_name  = self._model_var.get(),
                n_tta       = int(self._tta_var.get()),
                temperature = float(self._temp_var.get()),
                cam_method  = self._cam_var.get(),
            )
            self._result = result
            self.after(0, lambda: self._update(result))
        except Exception as e:
            self.after(0, lambda: self._on_err(str(e)))

    def _update(self, r):
        self._spinning = False
        self._spin_cv.delete("all")

        color = MAL_CLR if r.is_malignant else BEN_CLR
        bg    = "#1F0D0D" if r.is_malignant else "#0D1A10"

        # ── Linear meter ──────────────────────────────────────────────
        _draw_linear_meter(self._gauge_cv, r.confidence, color, r.n_tta_views)

        # ── Classification result card ────────────────────────────────
        self._verdict_var.set(r.verdict)
        self._verdict_lbl.config(fg=color)
        self._class_var.set(r.predicted_class)
        cls_color = CLASS_COLORS.get(r.predicted_class, TEXT2)
        self._class_lbl.config(fg=cls_color)

        # Uncertainty
        uc       = r.uncertainty
        uc_color = BEN_CLR if uc < 0.05 else WARN_CLR if uc < 0.12 else MAL_CLR
        self._unc_var.set(f"±{uc * 100:.2f}%")

        # ── Prob bars ─────────────────────────────────────────────────
        self._draw_probs(r.probabilities, highlight=r.predicted_class)

        # ── Images ────────────────────────────────────────────────────
        b64s = {
            "_img_orig": r.original_b64,
            "_img_heat": r.heatmap_b64,
            "_img_over": r.overlay_b64,
            "_img_tta":  r.panel_b64,
        }
        for key, b64 in b64s.items():
            if not b64:
                continue
            lbl = self._img_labels[key]
            w   = max(lbl.winfo_width()  - 8, 60)
            h   = max(lbl.winfo_height() - 8, 60)
            ph  = _b64_to_pil(b64, (w, h))
            if ph:
                lbl.config(image=ph)
                self._photos[key] = ph

        # ── Meta ──────────────────────────────────────────────────────
        self._meta["Model"].set(r.model_name)
        self._meta["TTA Views"].set(str(r.n_tta_views))
        self._meta["CAM Method"].set(self._cam_var.get())
        self._meta["Inference Time"].set(f"{r.inference_time_s:.2f}s")
        self._meta["Image Shape"].set(str(r.image_shape))

        # ── History ───────────────────────────────────────────────────
        fname = Path(self._img_path).name
        entry = f"{r.predicted_class} ({r.confidence * 100:.0f}%)  {fname}"
        self._history.append((entry, r))
        self._hist_box.insert("end", entry)
        self._hist_box.itemconfig("end",
                                   fg=MAL_CLR if r.is_malignant else BEN_CLR)

        self._run_btn.config(state="normal", text="▶   RUN ANALYSIS")
        self._status(
            f"✓  {r.predicted_class}  ({r.confidence * 100:.1f}%)  "
            f"|  ±{r.uncertainty * 100:.2f}%  |  {r.inference_time_s:.2f}s")

    def _on_err(self, msg: str):
        self._spinning = False
        self._run_btn.config(state="normal", text="▶   RUN ANALYSIS")
        self._verdict_var.set("ERROR")
        self._verdict_lbl.config(fg=MAL_CLR)
        self._status(f"✗  {msg}")
        messagebox.showerror("Inference Error", msg)

    # ── History ────────────────────────────────────────────────────────

    def _on_hist(self, event):
        sel = self._hist_box.curselection()
        if sel and sel[0] < len(self._history):
            _, r = self._history[sel[0]]
            self._result = r
            self._update(r)

    # ── Export ─────────────────────────────────────────────────────────

    def _export(self):
        if not self._result:
            messagebox.showinfo("No Result", "Run an analysis first.")
            return
        r    = self._result
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JSON", "*.json")],
            initialfile=f"CRC_Report_{r.predicted_class.replace(' ', '_')}.png")
        if not path:
            return
        if path.endswith(".json"):
            data = {
                "predicted_class": r.predicted_class,
                "verdict":         r.verdict,
                "confidence":      r.confidence,
                "uncertainty":     r.uncertainty,
                "probabilities":   r.probabilities,
                "model":           r.model_name,
                "tta_views":       r.n_tta_views,
                "inference_time":  r.inference_time_s,
            }
            Path(path).write_text(json.dumps(data, indent=2))
        else:
            self._save_png(r, path)
        messagebox.showinfo("Exported", f"Saved:\n{path}")
        self._status(f"Exported → {Path(path).name}")

    def _save_png(self, r, path: str):
        W, H = 1400, 860
        img  = Image.new("RGB", (W, H), "#0B0F14")
        d    = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 52], fill="#080C10")
        d.text((20, 14), "CRC-AI  Colorectal Cancer Diagnostic Report",
               fill=ACCENT)
        pw, ph2 = 330, 260
        coords  = [(20, 60), (360, 60), (20, 330), (360, 330)]
        b64s    = [r.original_b64, r.heatmap_b64,
                   r.overlay_b64,  r.panel_b64]
        labels  = ["Original", "Heatmap", "Overlay", "TTA Panel"]
        for (x, y), b64, lbl in zip(coords, b64s, labels):
            if b64:
                try:
                    pil = Image.open(
                        io.BytesIO(base64.b64decode(b64))).convert("RGB")
                    pil = pil.resize((pw, ph2), Image.LANCZOS)
                    img.paste(pil, (x, y))
                except Exception:
                    pass
            d.text((x, y + ph2 + 4), lbl, fill=TEXT2)
        rx  = 720
        clr = MAL_CLR if r.is_malignant else BEN_CLR
        d.text((rx, 70),  r.verdict,         fill=clr)
        d.text((rx, 120), r.predicted_class,  fill=TEXT1)
        d.text((rx, 150), f"Confidence:  {r.confidence * 100:.2f}%",   fill=TEXT1)
        d.text((rx, 175), f"Uncertainty: ±{r.uncertainty * 100:.2f}%", fill=WARN_CLR)
        d.text((rx, 215), "CLASS PROBABILITIES", fill=TEXT2)
        for i, cls in enumerate(CLASSES_LIST):
            p   = r.probabilities.get(cls, 0)
            col = CLASS_COLORS.get(cls, ACCENT)
            bx, bw, by = rx + 160, 380, 240 + i * 35
            d.text((rx, by + 6), cls, fill=TEXT2)
            d.rectangle([bx, by, bx + bw, by + 20], fill=BORDER_CLR)
            if int(p * bw) > 0:
                d.rectangle([bx, by, bx + int(p * bw), by + 20], fill=col)
            d.text((bx + bw + 8, by + 6), f"{p * 100:.1f}%", fill=TEXT1)
        d.rectangle([0, H - 30, W, H], fill="#080C10")
        d.text((20, H - 18),
               f"Model: {r.model_name}  |  TTA: {r.n_tta_views}  "
               f"|  Time: {r.inference_time_s:.2f}s  |  CRC-AI",
               fill=TEXT3)
        img.save(path, dpi=(300, 300))

    # ── Status ─────────────────────────────────────────────────────────

    def _status(self, msg: str):
        self._status_var.set(msg[:130])


# ─── Entry point ───────────────────────────────────────────────────────────

def launch():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = ClinicalGUI()
    app.mainloop()


if __name__ == "__main__":
    launch()
