"""
人脸表情包工具 v1.2
给图片中的人脸自动覆盖自定义表情包，支持多人脸、不同表情、自动旋转、缩放调节和拖拽微调。
使用 OpenCV DNN (YuNet) 人脸检测，比 Haar Cascade 更准，支持侧脸/暗光。

依赖: pip install opencv-python pillow
运行: python face_emoji.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import math
import os
import sys
from typing import Optional, Any

# ──────────────────────────────────────────────

class FaceEmojiApp:
    """主应用类：UI + 检测 + 合成 + 交互。"""

    # ── 常量 ──
    # 颜色
    COLOR_BG: str = '#2b2b2b'
    COLOR_SELECTED: str = '#00ccff'
    COLOR_NORMAL: str = '#00ff88'
    COLOR_TEXT_SECONDARY: str = '#999'
    COLOR_TEXT_HINT: str = '#555'
    COLOR_THUMB_BORDER: str = '#333'

    # 尺寸
    THUMB_SIZE: int = 60
    CANVAS_MIN_SIZE: int = 100
    CANVAS_MARGIN: int = 20
    MAX_FIT_SCALE: float = 2.0
    FACE_BBOX_MARGIN: int = 4
    FACE_CLICK_MARGIN: int = 10
    EMOJI_MIN_DIM: int = 4

    # 检测参数 — YuNet
    YUNET_SCORE_THRESHOLD: float = 0.7
    YUNET_NMS_THRESHOLD: float = 0.3
    YUNET_TOP_K: int = 5000
    MAX_ANGLE: float = 45.0

    # 滑块范围
    SLIDER_FROM: float = 0.3
    SLIDER_TO: float = 2.0

    # 交互参数
    DEBOUNCE_MS: int = 100
    NUDGE_PX: int = 2
    ZOOM_MIN: float = 0.1
    ZOOM_MAX: float = 10.0
    ZOOM_FACTOR: float = 1.15

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("人脸表情包工具 v1.2")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

        # ─── 数据 ───
        self.original_image: Optional[Image.Image] = None  # PIL RGBA 全分辨率
        self.image_path: Optional[str] = None
        self.faces_data: list[dict[str, Any]] = []         # 每人脸一条
        self.selected_idx: Optional[int] = None            # 当前选中索引
        self.drag_info: Optional[tuple[float, float, int, int]] = None

        # ─── 画布状态 ───
        self._canvas_w: int = 1
        self._canvas_h: int = 1
        self._img_on_canvas_scale: float = 1.0   # 原图 → 画布缩放比
        self._offset_x: float = 0                 # 画布上图片偏移（居中）
        self._offset_y: float = 0
        self._zoom_level: float = 1.0             # 缩放倍数（1.0 = 适应画布）
        self._canvas_img_id: Optional[int] = None
        self._bbox_ids: list[int] = []
        self._composite: Optional[Image.Image] = None  # 全分辨率合成图

        # ─── 防抖 ───
        self._rebuild_timer: Optional[str] = None

        # ─── OpenCV DNN 检测器 (YuNet) ───
        model_path: str = self._resolve_model_path('face_detection_yunet_2023mar.onnx')
        self.face_detector: cv2.FaceDetectorYN = cv2.FaceDetectorYN.create(
            model=model_path, config='',
            input_size=(320, 320),
            score_threshold=self.YUNET_SCORE_THRESHOLD,
            nms_threshold=self.YUNET_NMS_THRESHOLD,
            top_k=self.YUNET_TOP_K,
        )

        self._build_ui()
        self.root.update_idletasks()
        self._canvas_w = max(self.canvas.winfo_width(), self.CANVAS_MIN_SIZE)
        self._canvas_h = max(self.canvas.winfo_height(), self.CANVAS_MIN_SIZE)

    @staticmethod
    def _resolve_model_path(filename: str) -> str:
        """返回模型文件的完整路径（兼容开发环境和 PyInstaller 打包后）。"""
        if getattr(sys, 'frozen', False):
            return os.path.join(sys._MEIPASS, filename)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    # ════════════════ UI 构建 ════════════════

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ── 左侧：画布 ──
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(canvas_frame, bg=self.COLOR_BG,
                                highlightthickness=0,
                                xscrollcommand=h_scroll.set,
                                yscrollcommand=v_scroll.set)
        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── 右侧：控制面板 ──
        control = ttk.Frame(main_frame, width=280)
        control.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)
        control.pack_propagate(False)

        ttk.Button(control, text='📷 打开图片', command=self.open_image).pack(fill=tk.X, pady=2)
        ttk.Button(control, text='👤 检测人脸', command=self.detect_faces).pack(fill=tk.X, pady=2)

        ttk.Separator(control).pack(fill=tk.X, pady=4)
        ttk.Label(control, text='── 已检测到的人脸 ──').pack()

        # ── 缩略图 ──
        thumb_container = ttk.Frame(control)
        thumb_container.pack(fill=tk.X, pady=2)
        self.thumb_canvas = tk.Canvas(thumb_container, height=72, highlightthickness=0)
        thumb_h_scroll = ttk.Scrollbar(thumb_container, orient=tk.HORIZONTAL,
                                       command=self.thumb_canvas.xview)
        self.thumb_canvas.configure(xscrollcommand=thumb_h_scroll.set)
        self.thumb_canvas.pack(side=tk.TOP, fill=tk.X)
        thumb_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.thumb_inner = ttk.Frame(self.thumb_canvas)
        self.thumb_canvas.create_window((0, 0), window=self.thumb_inner, anchor='nw')
        self.thumb_inner.bind('<Configure>',
            lambda e: self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox('all')))

        self.selected_label = ttk.Label(control, text='当前选中: (无)')
        self.selected_label.pack(pady=2)
        ttk.Separator(control).pack(fill=tk.X, pady=4)

        # ── 操作按钮 ──
        btn_frame = ttk.Frame(control)
        btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text='😀 给此人加载', command=self.load_emoji).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text='👥 给所有人', command=self.load_emoji_to_all).pack(
            side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))
        ttk.Button(control, text='🗑 清除此人表情', command=self.clear_emoji).pack(fill=tk.X, pady=2)

        # ── 大小滑块 ──
        scale_frame = ttk.Frame(control)
        scale_frame.pack(fill=tk.X, pady=4)
        ttk.Label(scale_frame, text='大小:').pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_slider = ttk.Scale(
            scale_frame, from_=self.SLIDER_FROM, to=self.SLIDER_TO,
            variable=self.scale_var, orient=tk.HORIZONTAL,
            command=self._on_scale_change
        )
        self.scale_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.scale_val_label = ttk.Label(control, text='1.00×', anchor=tk.CENTER)
        self.scale_val_label.pack(fill=tk.X)

        self.offset_label = ttk.Label(control, text='偏移: (0, 0)')
        self.offset_label.pack()
        self.zoom_label = ttk.Label(control, text='缩放: 100%', foreground=self.COLOR_TEXT_SECONDARY)
        self.zoom_label.pack()

        ttk.Separator(control).pack(fill=tk.X, pady=4)
        ttk.Button(control, text='💾 保存图片', command=self.save_image).pack(fill=tk.X, pady=2)

        # ── 提示文字 ──
        self.info_text = tk.Text(control, height=6, width=28, fg=self.COLOR_TEXT_HINT,
                                 relief=tk.FLAT, state=tk.DISABLED, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X, pady=(8, 2))
        self._set_info('就绪\n\n打开图片后点击\n"检测人脸"开始')

        # ── 底部状态栏 ──
        self.status_var = tk.StringVar(value='就绪')
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # ── 事件绑定 ──
        self.canvas.bind('<Button-1>', self._on_canvas_click)
        self.canvas.bind('<B1-Motion>', self._on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_canvas_release)
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)       # Windows
        self.canvas.bind('<Button-4>', self._on_mouse_wheel_linux)   # Linux up
        self.canvas.bind('<Button-5>', self._on_mouse_wheel_linux)   # Linux down

        self.root.bind('<Tab>', lambda e: self._select_next_face())
        self.root.bind('<Left>', lambda e: self._nudge_offset(-self.NUDGE_PX, 0))
        self.root.bind('<Right>', lambda e: self._nudge_offset(self.NUDGE_PX, 0))
        self.root.bind('<Up>', lambda e: self._nudge_offset(0, -self.NUDGE_PX))
        self.root.bind('<Down>', lambda e: self._nudge_offset(0, self.NUDGE_PX))
        self.root.bind('<Escape>', lambda e: self._deselect_face())
        self.root.bind('<Control-plus>', lambda e: self._zoom_at_canvas_center(1))
        self.root.bind('<Control-minus>', lambda e: self._zoom_at_canvas_center(-1))

    def _set_info(self, text: str) -> None:
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', text)
        self.info_text.config(state=tk.DISABLED)

    # ════════════════ 图片加载 ════════════════

    def open_image(self) -> None:
        path: str = filedialog.askopenfilename(
            title='选择图片',
            filetypes=[('图片文件', '*.jpg *.jpeg *.png *.bmp *.webp'), ('所有文件', '*.*')]
        )
        if not path:
            return

        try:
            self.original_image = Image.open(path).convert('RGBA')
            self.image_path = path
            self.faces_data = []
            self.selected_idx = None
            self.drag_info = None
            self._zoom_level = 1.0

            self._canvas_w = max(self.canvas.winfo_width(), self.CANVAS_MIN_SIZE)
            self._canvas_h = max(self.canvas.winfo_height(), self.CANVAS_MIN_SIZE)
            self._rebuild_and_redraw()
            self._refresh_thumbnails()
            self._update_controls()
            self.status_var.set(f'已加载: {os.path.basename(path)}')
            self._set_info(f'加载成功\n{os.path.basename(path)}\n'
                           f'{self.original_image.width}×{self.original_image.height}\n\n'
                           f'点击"检测人脸"继续')
        except Exception as e:
            messagebox.showerror('错误', f'无法打开图片:\n{e}')

    def _on_canvas_resize(self, event: Optional[tk.Event] = None) -> None:
        if event:
            self._canvas_w = max(event.width, self.CANVAS_MIN_SIZE)
            self._canvas_h = max(event.height, self.CANVAS_MIN_SIZE)
        if self.original_image is not None:
            self._redraw_canvas()

    # ════════════════ 合成与绘制 ════════════════

    def _rebuild_and_redraw(self) -> None:
        """重建合成图并刷新画布。"""
        if self.original_image is None:
            return
        self._rebuild_composite()
        self._redraw_canvas()

    def _rebuild_composite(self) -> None:
        """把所有表情合成到原图上（全分辨率），结果存到 self._composite。"""
        comp: Image.Image = self.original_image.copy().convert('RGBA')

        for face in self.faces_data:
            emoji_img: Optional[Image.Image] = face.get('emoji_img')
            if emoji_img is None:
                continue

            emoji: Image.Image = emoji_img.convert('RGBA')
            orig_w, orig_h = emoji.size
            x, y, w, h = face['bbox']
            cx: int = x + w // 2
            cy: int = y + h // 2
            scale: float = face['scale']
            ox, oy = face['offset']

            # 以人脸较大边为基准，保持表情原始宽高比
            face_size: int = max(w, h)
            target: int = max(int(face_size * scale), self.EMOJI_MIN_DIM)
            if orig_w >= orig_h:
                ew: int = target
                eh: int = max(int(target * orig_h / orig_w), self.EMOJI_MIN_DIM)
            else:
                eh = target
                ew = max(int(target * orig_w / orig_h), self.EMOJI_MIN_DIM)
            emoji_resized: Image.Image = emoji.resize((ew, eh), Image.LANCZOS)

            angle: float = face.get('angle', 0)
            if abs(angle) > 0.5:
                emoji_resized = emoji_resized.rotate(
                    angle, expand=True, center=(ew / 2, eh / 2),
                    resample=Image.BICUBIC
                )

            paste_x: int = cx + ox - emoji_resized.width // 2
            paste_y: int = cy + oy - emoji_resized.height // 2

            if emoji_resized.mode == 'RGBA':
                comp.paste(emoji_resized, (paste_x, paste_y), emoji_resized)
            else:
                comp.paste(emoji_resized, (paste_x, paste_y))

        self._composite = comp

    def _redraw_canvas(self) -> None:
        """把合成图按当前缩放缩放到画布并绘制，同时画人脸框和编号。"""
        if self._composite is None:
            return

        cw, ch = self._canvas_w, self._canvas_h
        img_w, img_h = self._composite.size

        # 基础缩放（适应画布），再乘以用户缩放倍数
        margin = self.CANVAS_MARGIN
        base_scale: float = min((cw - margin * 2) / img_w,
                                (ch - margin * 2) / img_h, self.MAX_FIT_SCALE)
        scale: float = base_scale * self._zoom_level
        disp_w: int = max(int(img_w * scale), 1)
        disp_h: int = max(int(img_h * scale), 1)

        self._img_on_canvas_scale = scale
        self._offset_x = max(0.0, (cw - disp_w) // 2)
        self._offset_y = max(0.0, (ch - disp_h) // 2)

        # 缩放显示用图片
        disp_img: Image.Image = self._composite.resize((disp_w, disp_h), Image.LANCZOS)
        self._tk_img: ImageTk.PhotoImage = ImageTk.PhotoImage(disp_img)

        self.canvas.delete('all')
        self._canvas_img_id = self.canvas.create_image(
            self._offset_x, self._offset_y, anchor='nw', image=self._tk_img
        )

        # 可滚动区域（图片区域 + 边距）
        scroll_w: int = max(cw, disp_w + int(self._offset_x * 2))
        scroll_h: int = max(ch, disp_h + int(self._offset_y * 2))
        self.canvas.configure(scrollregion=(0, 0, scroll_w, scroll_h))

        # 画人脸框
        self._bbox_ids = []
        m = self.FACE_BBOX_MARGIN
        for i, face in enumerate(self.faces_data):
            x, y, w, h = face['bbox']
            x1: int = int((x - m) * scale + self._offset_x)
            y1: int = int((y - m) * scale + self._offset_y)
            x2: int = int((x + w + m) * scale + self._offset_x)
            y2: int = int((y + h + m) * scale + self._offset_y)

            is_selected: bool = (i == self.selected_idx)
            outline: str = self.COLOR_SELECTED if is_selected else self.COLOR_NORMAL
            width: int = 2 if is_selected else 1

            bid: int = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=outline, width=width, dash=() if is_selected else (4, 3)
            )
            self._bbox_ids.append(bid)

            label: str = str(i + 1)
            if face.get('emoji_img') is not None:
                label += ' 😀'
            label_y: int = y1 - 14 if y1 > 14 else y1 + 3
            self.canvas.create_text(x1 + 3, label_y, text=label, anchor='nw',
                                    fill=outline, font=('Arial', 10, 'bold'))

        # 更新缩放标签
        zoom_pct: int = round(self._zoom_level * 100)
        self.zoom_label.config(text=f'缩放: {zoom_pct}%  (滚轮/Ctrl+±)')

    # ════════════════ 防抖 ════════════════

    def _schedule_rebuild(self, delay_ms: int = DEBOUNCE_MS) -> None:
        """防抖调度：取消上次待执行的合成，在 delay_ms 后执行。"""
        if self._rebuild_timer:
            self.root.after_cancel(self._rebuild_timer)
        self._rebuild_timer = self.root.after(delay_ms, self._do_debounced_rebuild)

    def _do_debounced_rebuild(self) -> None:
        self._rebuild_timer = None
        self._rebuild_and_redraw()

    # ════════════════ 人脸检测 ════════════════

    def detect_faces(self) -> None:
        if self.original_image is None:
            messagebox.showinfo('提示', '请先打开一张图片')
            return

        self.status_var.set('人脸检测中...')
        self.root.update_idletasks()

        try:
            img_rgb: np.ndarray = np.array(self.original_image.convert('RGB'))
            img_bgr: np.ndarray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            height, width = img_bgr.shape[:2]

            # YuNet DNN 检测
            self.face_detector.setInputSize((width, height))
            _, faces = self.face_detector.detect(img_bgr)

            if faces is None or len(faces) == 0:
                self.status_var.set('未检测到人脸')
                self._set_info('未检测到人脸 😅\n\n试试换个角度或\n更清晰的照片')
                return

            self.faces_data = []
            for i in range(faces.shape[0]):
                face_data = faces[i]
                # YuNet 输出: [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y,
                #               rmouth_x, rmouth_y, lmouth_x, lmouth_y, confidence]
                fx, fy, fw, fh = face_data[:4].astype(int)

                # 从双眼关键点计算角度（YuNet 比 Haar 可靠得多）
                le_x, le_y = face_data[6], face_data[7]   # 左眼
                re_x, re_y = face_data[4], face_data[5]   # 右眼
                dx: float = le_x - re_x
                dy: float = le_y - re_y
                if abs(dx) > 0:
                    angle: float = math.degrees(math.atan2(dy, dx))
                    if abs(angle) > self.MAX_ANGLE:
                        angle = 0.0
                else:
                    angle = 0.0

                # 裁剪缩略图
                roi: Image.Image = self.original_image.crop((fx, fy, fx + fw, fy + fh))
                thumb: Image.Image = roi.resize((self.THUMB_SIZE, self.THUMB_SIZE), Image.LANCZOS)
                thumb_tk: ImageTk.PhotoImage = ImageTk.PhotoImage(thumb)

                self.faces_data.append({
                    'id': i,
                    'bbox': (int(fx), int(fy), int(fw), int(fh)),
                    'angle': angle,
                    'emoji_img': None,
                    'emoji_path': '',
                    'scale': 1.0,
                    'offset': (0, 0),
                    'thumb_pil': thumb,
                    'thumb_tk': thumb_tk,
                })

            self.selected_idx = 0
            self._sync_slider_to_selected()
            self._rebuild_and_redraw()
            self._refresh_thumbnails()
            self._update_controls()

            emoji_count: int = sum(1 for f in self.faces_data if f['emoji_img'] is not None)
            self.status_var.set(f'检测到 {len(self.faces_data)} 张人脸，已分配表情: {emoji_count}')
            self._set_info(
                f'✅ 检测到 {len(self.faces_data)} 张人脸\n\n'
                f'• 点击人脸选中\n'
                f'• 点击"加载表情"分配\n'
                f'• 拖拽可微调位置'
            )
        except Exception as e:
            messagebox.showerror('错误', f'人脸检测失败:\n{e}')
            self.status_var.set('检测出错')

    # ════════════════ 缩略图 ════════════════

    def _refresh_thumbnails(self) -> None:
        for w in self.thumb_inner.winfo_children():
            w.destroy()

        if not self.faces_data:
            lbl = ttk.Label(self.thumb_inner, text='(无)', foreground=self.COLOR_TEXT_SECONDARY)
            lbl.pack(side=tk.LEFT, padx=4)
            return

        for i, face in enumerate(self.faces_data):
            container = ttk.Frame(self.thumb_inner)

            if i == self.selected_idx:
                border_frame = tk.Frame(container, highlightbackground=self.COLOR_SELECTED,
                                        highlightthickness=2, bd=0)
            else:
                border_frame = tk.Frame(container, highlightbackground=self.COLOR_THUMB_BORDER,
                                        highlightthickness=1, bd=0)

            img_label = ttk.Label(border_frame, image=face['thumb_tk'])
            img_label.pack()
            border_frame.pack(side=tk.LEFT, padx=3, pady=2)

            num_label = ttk.Label(container, text=str(i + 1),
                                  font=('Arial', 8), foreground=self.COLOR_TEXT_SECONDARY)
            num_label.pack()
            container.pack(side=tk.LEFT, padx=2)

            container.bind('<Button-1>', lambda e, idx=i: self._select_face(idx))
            border_frame.bind('<Button-1>', lambda e, idx=i: self._select_face(idx))
            img_label.bind('<Button-1>', lambda e, idx=i: self._select_face(idx))
            num_label.bind('<Button-1>', lambda e, idx=i: self._select_face(idx))

        self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox('all'))

    def _select_face(self, idx: int) -> None:
        if 0 <= idx < len(self.faces_data):
            self.selected_idx = idx
            self.drag_info = None
            self._sync_slider_to_selected()
            self._rebuild_and_redraw()
            self._refresh_thumbnails()
            self._update_controls()
            self.status_var.set(f'已选中 人脸 #{idx + 1}')

    def _select_next_face(self) -> str:
        if not self.faces_data:
            return 'break'
        if self.selected_idx is None:
            self._select_face(0)
        else:
            self._select_face((self.selected_idx + 1) % len(self.faces_data))
        return 'break'

    def _deselect_face(self) -> None:
        self.selected_idx = None
        self.drag_info = None
        self._redraw_canvas()
        self._refresh_thumbnails()
        self._update_controls()

    # ════════════════ 画布交互 ════════════════

    def _canvas_to_img(self, event_x: float, event_y: float
                       ) -> tuple[float, float]:
        """画布事件坐标 → 原图坐标（考虑滚动和缩放）。"""
        cx: float = self.canvas.canvasx(event_x)
        cy: float = self.canvas.canvasy(event_y)
        img_x: float = (cx - self._offset_x) / self._img_on_canvas_scale
        img_y: float = (cy - self._offset_y) / self._img_on_canvas_scale
        return img_x, img_y

    def _on_canvas_click(self, event: tk.Event) -> None:
        if not self.faces_data:
            return

        img_x, img_y = self._canvas_to_img(event.x, event.y)

        clicked: Optional[int] = None
        for i, face in enumerate(self.faces_data):
            x, y, w, h = face['bbox']
            margin = self.FACE_CLICK_MARGIN
            if (x - margin) <= img_x <= (x + w + margin) and \
               (y - margin) <= img_y <= (y + h + margin):
                clicked = i
                break

        if clicked is not None:
            self._select_face(clicked)
            face = self.faces_data[clicked]
            # 存画布坐标（含滚动偏移）
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            self.drag_info = (cx, cy, face['offset'][0], face['offset'][1])
        else:
            self._deselect_face()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self.selected_idx is None or self.drag_info is None:
            return

        start_cx, start_cy, orig_ox, orig_oy = self.drag_info
        cx: float = self.canvas.canvasx(event.x)
        cy: float = self.canvas.canvasy(event.y)
        dx: float = (cx - start_cx) / self._img_on_canvas_scale
        dy: float = (cy - start_cy) / self._img_on_canvas_scale

        face = self.faces_data[self.selected_idx]
        face['offset'] = (round(orig_ox + dx), round(orig_oy + dy))

        self._update_controls()
        self._schedule_rebuild(80)  # 防抖 80ms

    def _on_canvas_release(self, event: tk.Event) -> None:
        # 如果防抖还没触发，立即重建
        if self.drag_info is not None and self.selected_idx is not None:
            self._do_debounced_rebuild()
        self.drag_info = None

    def _nudge_offset(self, dx: int, dy: int) -> None:
        if self.selected_idx is None:
            return
        face = self.faces_data[self.selected_idx]
        ox, oy = face['offset']
        face['offset'] = (ox + dx, oy + dy)
        self._rebuild_and_redraw()
        self._update_controls()

    # ════════════════ 滚轮缩放 ════════════════

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        """Windows 滚轮事件。"""
        # event.delta: +120 (上), -120 (下)
        direction: int = 1 if event.delta > 0 else -1
        self._zoom_at_cursor(event.x, event.y, direction)

    def _on_mouse_wheel_linux(self, event: tk.Event) -> None:
        """Linux 滚轮事件（Button-4/5）。"""
        direction: int = 1 if event.num == 4 else -1
        self._zoom_at_cursor(event.x, event.y, direction)

    def _zoom_at_cursor(self, viewport_x: float, viewport_y: float, direction: int) -> None:
        """以鼠标位置为中心缩放。"""
        # 缩放前鼠标位置的画布坐标
        pre_cx: float = self.canvas.canvasx(viewport_x)
        pre_cy: float = self.canvas.canvasy(viewport_y)

        old_zoom: float = self._zoom_level
        if direction > 0:
            self._zoom_level *= self.ZOOM_FACTOR
        else:
            self._zoom_level /= self.ZOOM_FACTOR
        self._zoom_level = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom_level))

        self._redraw_canvas()

        # 缩放后把鼠标下的点保持在视口同一位置
        post_cx: float = pre_cx * (self._zoom_level / old_zoom)
        post_cy: float = pre_cy * (self._zoom_level / old_zoom)
        self.canvas.xview_moveto((post_cx - viewport_x) / self.canvas.bbox('all')[2]
                                 if self.canvas.bbox('all') else 0)
        self.canvas.yview_moveto((post_cy - viewport_y) / self.canvas.bbox('all')[3]
                                 if self.canvas.bbox('all') else 0)

    def _zoom_at_canvas_center(self, direction: int) -> None:
        """以画布中心缩放（Ctrl+±快捷键）。"""
        cx = self._canvas_w // 2
        cy = self._canvas_h // 2
        self._zoom_at_cursor(cx, cy, direction)

    # ════════════════ 表情操作 ════════════════

    def load_emoji(self) -> None:
        if self.selected_idx is None:
            messagebox.showinfo('提示', '请先选中一张人脸')
            return

        path: str = filedialog.askopenfilename(
            title='选择表情图片',
            filetypes=[('图片文件', '*.png *.jpg *.jpeg *.webp *.bmp'), ('所有文件', '*.*')]
        )
        if not path:
            return

        try:
            emoji: Image.Image = Image.open(path).convert('RGBA')
            face = self.faces_data[self.selected_idx]
            face['emoji_img'] = emoji
            face['emoji_path'] = path
            face['scale'] = self.scale_var.get()

            self._rebuild_and_redraw()
            self._refresh_thumbnails()
            self._update_controls()

            emoji_count: int = sum(1 for f in self.faces_data if f['emoji_img'] is not None)
            self.status_var.set(
                f'已为人脸 #{self.selected_idx + 1} 加载表情 ({emoji_count}/{len(self.faces_data)})')
            self._set_info(
                f'✅ 人脸 #{self.selected_idx + 1} 已加载\n'
                f'{os.path.basename(path)}\n\n'
                f'拖动滑块调大小\n拖拽表情微调位置'
            )
        except Exception as e:
            messagebox.showerror('错误', f'无法加载表情:\n{e}')

    def load_emoji_to_all(self) -> None:
        """给所有检测到的人脸一次性加载相同表情（保持各人脸已有偏移不变）。"""
        if not self.faces_data:
            messagebox.showinfo('提示', '请先检测人脸')
            return

        path: str = filedialog.askopenfilename(
            title='选择表情图片（将应用到所有人脸）',
            filetypes=[('图片文件', '*.png *.jpg *.jpeg *.webp *.bmp'), ('所有文件', '*.*')]
        )
        if not path:
            return

        try:
            emoji: Image.Image = Image.open(path).convert('RGBA')
            current_scale: float = self.scale_var.get()

            for face in self.faces_data:
                face['emoji_img'] = emoji.copy()
                face['emoji_path'] = path
                face['scale'] = current_scale
                # ⚠ 不再重置 offset — 保留各人脸已有偏移

            self.selected_idx = 0
            self._sync_slider_to_selected()
            self._rebuild_and_redraw()
            self._refresh_thumbnails()
            self._update_controls()

            self.status_var.set(f'已为全部 {len(self.faces_data)} 张人脸加载相同表情')
            self._set_info(
                f'✅ 已批量应用到 {len(self.faces_data)} 张人脸\n\n'
                f'每张人脸可单独选中调大小\n或点"给此人加载"换不同的'
            )
        except Exception as e:
            messagebox.showerror('错误', f'无法加载表情:\n{e}')

    def clear_emoji(self) -> None:
        if self.selected_idx is None:
            return
        face = self.faces_data[self.selected_idx]
        if face['emoji_img'] is None:
            return

        face['emoji_img'] = None
        face['emoji_path'] = ''
        face['scale'] = 1.0
        face['offset'] = (0, 0)
        self.scale_var.set(1.0)

        self._rebuild_and_redraw()
        self._refresh_thumbnails()
        self._update_controls()
        self.status_var.set(f'已清除 人脸 #{self.selected_idx + 1} 的表情')

    # ════════════════ 滑块 ════════════════

    def _on_scale_change(self, value: str) -> None:
        """滑块拖动中：只更新数据和标签，延迟重绘。"""
        if self.selected_idx is not None:
            face = self.faces_data[self.selected_idx]
            face['scale'] = float(value)
        self.scale_val_label.config(text=f'{float(value):.2f}×')
        self._schedule_rebuild()

    def _sync_slider_to_selected(self) -> None:
        if self.selected_idx is not None:
            face = self.faces_data[self.selected_idx]
            self.scale_var.set(face['scale'])
            self.scale_val_label.config(text=f'{face["scale"]:.2f}×')

    # ════════════════ UI 状态 ════════════════

    def _update_controls(self) -> None:
        if self.selected_idx is not None:
            face = self.faces_data[self.selected_idx]
            has_emoji: bool = face['emoji_img'] is not None
            ox, oy = face['offset']
            self.selected_label.config(
                text=f'当前选中: 人脸 #{self.selected_idx + 1}' +
                     (' 😀' if has_emoji else '')
            )
            self.offset_label.config(text=f'偏移: ({ox}, {oy})')
        else:
            self.selected_label.config(text='当前选中: (无)')
            self.offset_label.config(text='偏移: (0, 0)')

    # ════════════════ 保存 ════════════════

    def save_image(self) -> None:
        if self._composite is None:
            messagebox.showinfo('提示', '没有可保存的内容')
            return

        has_any: bool = any(f.get('emoji_img') is not None for f in self.faces_data)
        if not has_any:
            if not messagebox.askyesno('确认', '还没有加载任何表情，确定要保存吗？'):
                return

        path: str = filedialog.asksaveasfilename(
            title='保存图片',
            defaultextension='.png',
            filetypes=[('PNG 图片', '*.png'), ('JPEG 图片', '*.jpg'), ('所有文件', '*.*')]
        )
        if not path:
            return

        try:
            ext: str = os.path.splitext(path)[1].lower()
            save_img: Image.Image = self._composite
            if ext in ('.jpg', '.jpeg'):
                save_img = save_img.convert('RGB')
            save_img.save(path)
            self.status_var.set(f'已保存: {os.path.basename(path)}')
            messagebox.showinfo('完成', f'图片已保存到:\n{path}')
        except Exception as e:
            messagebox.showerror('错误', f'保存失败:\n{e}')


# ──────────────────────────────────────────────

if __name__ == '__main__':
    root: tk.Tk = tk.Tk()
    app: FaceEmojiApp = FaceEmojiApp(root)
    root.mainloop()
