"""
人脸表情包工具 v1.0
给图片中的人脸自动覆盖自定义表情包，支持逐人分配不同表情、自动旋转对齐、大小调节和拖拽微调。

依赖: pip install opencv-python
运行: python face_emoji.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import math
import os

# ──────────────────────────────────────────────

class FaceEmojiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸表情包工具 v1.0")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

        # ─── 数据 ───
        self.original_image = None       # PIL Image (RGBA, 全分辨率)
        self.image_path = None
        self.faces_data = []             # 每张人脸的数据
        self.selected_idx = None         # 当前选中的人脸索引
        self.drag_info = None            # (start_cx, start_cy, orig_ox, orig_oy)

        # ─── 画布显示状态 ───
        self._canvas_w = 1
        self._canvas_h = 1
        self._img_on_canvas_scale = 1.0  # 原图 → 画布缩放比
        self._offset_x = 0               # 画布上图片的偏移（居中）
        self._offset_y = 0
        self._canvas_img_id = None       # canvas 上的图片 item id
        self._bbox_ids = []              # canvas 上的人脸框 item id 列表

        # ─── OpenCV 检测器 ───
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        self._build_ui()
        self.root.update_idletasks()
        self._canvas_w = max(self.canvas.winfo_width(), 100)
        self._canvas_h = max(self.canvas.winfo_height(), 100)

    # ──────────────── UI 构建 ────────────────

    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ── 左侧：画布 ──
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(canvas_frame, bg='#2b2b2b',
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

        # ── 缩略图容器（带横向滚动） ──
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

        # ── 选中信息 ──
        self.selected_label = ttk.Label(control, text='当前选中: (无)')
        self.selected_label.pack(pady=2)

        ttk.Separator(control).pack(fill=tk.X, pady=4)

        # ── 操作按钮 ──
        btn_frame = ttk.Frame(control)
        btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text='😀 给此人加载', command=self.load_emoji).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text='👥 给所有人', command=self.load_emoji_to_all).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))
        ttk.Button(control, text='🗑 清除此人表情', command=self.clear_emoji).pack(fill=tk.X, pady=2)

        # ── 大小滑块 ──
        scale_frame = ttk.Frame(control)
        scale_frame.pack(fill=tk.X, pady=4)
        ttk.Label(scale_frame, text='大小:').pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_slider = ttk.Scale(
            scale_frame, from_=0.3, to=2.0, variable=self.scale_var,
            orient=tk.HORIZONTAL, command=self._on_scale_change
        )
        self.scale_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.scale_val_label = ttk.Label(control, text='1.00×', anchor=tk.CENTER)
        self.scale_val_label.pack(fill=tk.X)

        # ── 偏移信息 ──
        self.offset_label = ttk.Label(control, text='偏移: (0, 0)')
        self.offset_label.pack()

        ttk.Separator(control).pack(fill=tk.X, pady=4)
        ttk.Button(control, text='💾 保存图片', command=self.save_image).pack(fill=tk.X, pady=2)

        # ── 提示文字 ──
        self.info_text = tk.Text(control, height=6, width=28, fg='#555',
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
        self.root.bind('<Tab>', self._select_next_face)
        self.root.bind('<Left>', lambda e: self._nudge_offset(-2, 0))
        self.root.bind('<Right>', lambda e: self._nudge_offset(2, 0))
        self.root.bind('<Up>', lambda e: self._nudge_offset(0, -2))
        self.root.bind('<Down>', lambda e: self._nudge_offset(0, 2))
        self.root.bind('<Escape>', lambda e: self._deselect_face())

    def _set_info(self, text):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', text)
        self.info_text.config(state=tk.DISABLED)

    # ──────────────── 图片加载与显示 ────────────────

    def open_image(self):
        path = filedialog.askopenfilename(
            title='选择图片',
            filetypes=[
                ('图片文件', '*.jpg *.jpeg *.png *.bmp *.webp'),
                ('所有文件', '*.*')
            ]
        )
        if not path:
            return

        try:
            self.original_image = Image.open(path).convert('RGBA')
            self.image_path = path
            self.faces_data = []
            self.selected_idx = None
            self.drag_info = None

            self._canvas_w = max(self.canvas.winfo_width(), 100)
            self._canvas_h = max(self.canvas.winfo_height(), 100)
            self._rebuild_and_redraw()
            self._refresh_thumbnails()
            self._update_controls()
            self.status_var.set(f'已加载: {os.path.basename(path)}')
            self._set_info(f'加载成功\n{os.path.basename(path)}\n{self.original_image.width}×{self.original_image.height}\n\n点击"检测人脸"继续')
        except Exception as e:
            messagebox.showerror('错误', f'无法打开图片:\n{e}')

    def _on_canvas_resize(self, event=None):
        if event:
            self._canvas_w = max(event.width, 100)
            self._canvas_h = max(event.height, 100)
        if self.original_image is not None:
            self._redraw_canvas()

    def _rebuild_and_redraw(self):
        """重建合成图并刷新画布显示。"""
        if self.original_image is None:
            return
        self._rebuild_composite()
        self._redraw_canvas()

    def _rebuild_composite(self):
        """把所有表情合成到原图上（全分辨率），结果存到 self._composite。"""
        comp = self.original_image.copy().convert('RGBA')

        for face in self.faces_data:
            emoji_img = face.get('emoji_img')
            if emoji_img is None:
                continue

            emoji = emoji_img.convert('RGBA')
            orig_w, orig_h = emoji.size
            x, y, w, h = face['bbox']
            cx = x + w // 2
            cy = y + h // 2
            scale = face['scale']
            ox, oy = face['offset']

            # 以人脸较大边为基准，保持表情原始宽高比
            face_size = max(w, h)
            target = max(int(face_size * scale), 4)
            if orig_w >= orig_h:
                ew = target
                eh = max(int(target * orig_h / orig_w), 4)
            else:
                eh = target
                ew = max(int(target * orig_w / orig_h), 4)
            emoji_resized = emoji.resize((ew, eh), Image.LANCZOS)

            angle = face.get('angle', 0)
            if abs(angle) > 0.5:
                emoji_resized = emoji_resized.rotate(
                    angle, expand=True, center=(ew / 2, eh / 2),
                    resample=Image.BICUBIC
                )

            paste_x = cx + ox - emoji_resized.width // 2
            paste_y = cy + oy - emoji_resized.height // 2

            if emoji_resized.mode == 'RGBA':
                comp.paste(emoji_resized, (paste_x, paste_y), emoji_resized)
            else:
                comp.paste(emoji_resized, (paste_x, paste_y))

        self._composite = comp

    def _redraw_canvas(self):
        """把合成图缩放到画布尺寸并绘制，同时画人脸框。"""
        if not hasattr(self, '_composite') or self._composite is None:
            return

        cw, ch = self._canvas_w, self._canvas_h
        img_w, img_h = self._composite.size

        # 计算缩放（等比例，留边距）
        margin = 20
        avail_w = cw - margin * 2
        avail_h = ch - margin * 2
        scale = min(avail_w / img_w, avail_h / img_h, 2.0)  # 最大 2x 防止太模糊
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)

        self._img_on_canvas_scale = scale
        self._offset_x = (cw - disp_w) // 2
        self._offset_y = (ch - disp_h) // 2

        # 缩略显示用图片
        disp_img = self._composite.resize((disp_w, disp_h), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(disp_img)

        # 清空画布重绘
        self.canvas.delete('all')
        self._canvas_img_id = self.canvas.create_image(
            self._offset_x, self._offset_y, anchor='nw', image=self._tk_img
        )

        # 设置可滚动区域
        self.canvas.configure(scrollregion=(
            0, 0, disp_w + self._offset_x * 2, disp_h + self._offset_y * 2
        ))

        # 画人脸框
        self._bbox_ids = []
        margin_px = 4  # 框比实际人脸稍大一点
        for i, face in enumerate(self.faces_data):
            x, y, w, h = face['bbox']
            # 在画布坐标系中的位置
            x1 = (x - margin_px) * scale + self._offset_x
            y1 = (y - margin_px) * scale + self._offset_y
            x2 = (x + w + margin_px) * scale + self._offset_x
            y2 = (y + h + margin_px) * scale + self._offset_y

            is_selected = (i == self.selected_idx)
            outline = '#00ccff' if is_selected else '#00ff88'
            width = 2 if is_selected else 1

            bid = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=outline, width=width, dash=(4, 3) if not is_selected else ()
            )
            self._bbox_ids.append(bid)

            # 显示人脸编号
            label_text = str(i + 1)
            if face.get('emoji_img') is not None:
                label_text += ' 😀'
            self.canvas.create_text(x1 + 3, y1 - 14 if y1 > 14 else y1 + 3,
                                    text=label_text, anchor='nw',
                                    fill='#00ccff' if is_selected else '#00ff88',
                                    font=('Arial', 10, 'bold'))

    # ──────────────── 人脸检测 ────────────────

    def detect_faces(self):
        if self.original_image is None:
            messagebox.showinfo('提示', '请先打开一张图片')
            return

        self.status_var.set('人脸检测中...')
        self.root.update_idletasks()

        try:
            # OpenCV 用 BGR
            img_rgb = np.array(self.original_image.convert('RGB'))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) == 0:
                self.status_var.set('未检测到人脸')
                self._set_info('未检测到人脸 😅\n\n试试换个角度或\n更清晰的照片')
                return

            self.faces_data = []
            for i, (fx, fy, fw, fh) in enumerate(faces):
                # 检测旋转角度
                angle = self._detect_angle(gray, (fx, fy, fw, fh))

                # 生成缩略图
                roi = self.original_image.crop((fx, fy, fx + fw, fy + fh))
                thumb = roi.resize((60, 60), Image.LANCZOS)
                thumb_tk = ImageTk.PhotoImage(thumb)

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

            # 默认选中第一张人脸
            self.selected_idx = 0
            self._sync_slider_to_selected()
            self._rebuild_and_redraw()
            self._refresh_thumbnails()
            self._update_controls()

            emoji_count = sum(1 for f in self.faces_data if f['emoji_img'] is not None)
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

    def _detect_angle(self, gray_img, face_bbox):
        """通过双眼位置计算人脸倾斜角度。"""
        fx, fy, fw, fh = face_bbox
        face_roi = gray_img[fy:fy + fh, fx:fx + fw]
        if face_roi.size == 0:
            return 0.0

        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3, minSize=(20, 20))

        if len(eyes) >= 2:
            # 取 x 方向最左和最右的两个眼睛
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            left = eyes_sorted[0]
            right = eyes_sorted[-1]

            lx = fx + left[0] + left[2] // 2
            ly = fy + left[1] + left[3] // 2
            rx = fx + right[0] + right[2] // 2
            ry = fy + right[1] + right[3] // 2

            dx = rx - lx
            dy = ry - ly
            if dx != 0:
                angle = math.degrees(math.atan2(dy, dx))
                # 如果角度偏离水平太离谱（> 45°），可能是误检
                if abs(angle) > 45:
                    return 0.0
                return angle

        return 0.0

    # ──────────────── 缩略图 ────────────────

    def _refresh_thumbnails(self):
        for w in self.thumb_inner.winfo_children():
            w.destroy()

        if not self.faces_data:
            lbl = ttk.Label(self.thumb_inner, text='(无)', foreground='#999')
            lbl.pack(side=tk.LEFT, padx=4)
            return

        for i, face in enumerate(self.faces_data):
            container = ttk.Frame(self.thumb_inner)

            if i == self.selected_idx:
                border_frame = tk.Frame(container, highlightbackground='#00ccff',
                                        highlightthickness=2, bd=0)
            else:
                border_frame = tk.Frame(container, highlightbackground='#333',
                                        highlightthickness=1, bd=0)

            img_label = ttk.Label(border_frame, image=face['thumb_tk'])
            img_label.pack()
            border_frame.pack(side=tk.LEFT, padx=3, pady=2)

            num_label = ttk.Label(container, text=str(i + 1),
                                  font=('Arial', 8), foreground='#999')
            num_label.pack()

            container.pack(side=tk.LEFT, padx=2)

            # 点击事件
            container.bind('<Button-1>', lambda e, idx=i: self._select_face(idx))
            border_frame.bind('<Button-1>', lambda e, idx=i: self._select_face(idx))
            img_label.bind('<Button-1>', lambda e, idx=i: self._select_face(idx))
            num_label.bind('<Button-1>', lambda e, idx=i: self._select_face(idx))

        self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox('all'))

    def _select_face(self, idx):
        if 0 <= idx < len(self.faces_data):
            self.selected_idx = idx
            self.drag_info = None
            self._sync_slider_to_selected()
            self._rebuild_and_redraw()
            self._refresh_thumbnails()
            self._update_controls()
            self.status_var.set(f'已选中 人脸 #{idx + 1}')

    def _select_next_face(self, event=None):
        if not self.faces_data:
            return
        if self.selected_idx is None:
            self._select_face(0)
        else:
            self._select_face((self.selected_idx + 1) % len(self.faces_data))
        return 'break'

    def _deselect_face(self):
        self.selected_idx = None
        self.drag_info = None
        self._redraw_canvas()
        self._refresh_thumbnails()
        self._update_controls()

    # ──────────────── 画布交互 ────────────────

    def _on_canvas_click(self, event):
        if not self.faces_data:
            return

        # 转换画布坐标 → 原图坐标
        img_x = (event.x - self._offset_x) / self._img_on_canvas_scale
        img_y = (event.y - self._offset_y) / self._img_on_canvas_scale

        # 判断点击到了哪个人脸（从后往前，优先上层）
        clicked = None
        for i, face in enumerate(self.faces_data):
            x, y, w, h = face['bbox']
            margin = 10  # 点击容差
            if (x - margin) <= img_x <= (x + w + margin) and (y - margin) <= img_y <= (y + h + margin):
                clicked = i
                break

        if clicked is not None:
            self._select_face(clicked)
            # 记录拖拽起始
            face = self.faces_data[clicked]
            self.drag_info = (event.x, event.y, face['offset'][0], face['offset'][1])
        else:
            self._deselect_face()

    def _on_canvas_drag(self, event):
        if self.selected_idx is None or self.drag_info is None:
            return

        start_x, start_y, orig_ox, orig_oy = self.drag_info
        # 拖拽距离需要转换到原图坐标系
        dx = (event.x - start_x) / self._img_on_canvas_scale
        dy = (event.y - start_y) / self._img_on_canvas_scale

        face = self.faces_data[self.selected_idx]
        face['offset'] = (round(orig_ox + dx), round(orig_oy + dy))

        self._rebuild_and_redraw()
        self._update_controls()

    def _on_canvas_release(self, event):
        self.drag_info = None

    def _nudge_offset(self, dx, dy):
        if self.selected_idx is None:
            return
        face = self.faces_data[self.selected_idx]
        ox, oy = face['offset']
        face['offset'] = (ox + dx, oy + dy)
        self._rebuild_and_redraw()
        self._update_controls()

    # ──────────────── 表情操作 ────────────────

    def load_emoji(self):
        if self.selected_idx is None:
            messagebox.showinfo('提示', '请先选中一张人脸')
            return

        path = filedialog.askopenfilename(
            title='选择表情图片',
            filetypes=[
                ('图片文件', '*.png *.jpg *.jpeg *.webp *.bmp'),
                ('所有文件', '*.*')
            ]
        )
        if not path:
            return

        try:
            emoji = Image.open(path).convert('RGBA')
            face = self.faces_data[self.selected_idx]
            face['emoji_img'] = emoji
            face['emoji_path'] = path
            face['scale'] = self.scale_var.get()

            self._rebuild_and_redraw()
            self._refresh_thumbnails()
            self._update_controls()

            emoji_count = sum(1 for f in self.faces_data if f['emoji_img'] is not None)
            self.status_var.set(f'已为人脸 #{self.selected_idx + 1} 加载表情 ({emoji_count}/{len(self.faces_data)})')
            self._set_info(
                f'✅ 人脸 #{self.selected_idx + 1} 已加载\n'
                f'{os.path.basename(path)}\n\n'
                f'拖动滑块调大小\n'
                f'拖拽表情微调位置'
            )
        except Exception as e:
            messagebox.showerror('错误', f'无法加载表情:\n{e}')

    def load_emoji_to_all(self):
        """给所有检测到的人脸一次性加载相同表情。"""
        if not self.faces_data:
            messagebox.showinfo('提示', '请先检测人脸')
            return

        path = filedialog.askopenfilename(
            title='选择表情图片（将应用到所有人脸）',
            filetypes=[
                ('图片文件', '*.png *.jpg *.jpeg *.webp *.bmp'),
                ('所有文件', '*.*')
            ]
        )
        if not path:
            return

        try:
            emoji = Image.open(path).convert('RGBA')
            current_scale = self.scale_var.get()

            for face in self.faces_data:
                face['emoji_img'] = emoji.copy()
                face['emoji_path'] = path
                face['scale'] = current_scale
                face['offset'] = (0, 0)

            # 回到第一张人脸
            self.selected_idx = 0
            self._sync_slider_to_selected()
            self._rebuild_and_redraw()
            self._refresh_thumbnails()
            self._update_controls()

            self.status_var.set(f'已为全部 {len(self.faces_data)} 张人脸加载相同表情')
            self._set_info(
                f'✅ 已批量应用到 {len(self.faces_data)} 张人脸\n\n'
                f'每张人脸可单独选中调大小\n'
                f'或继续点"给此人加载"换不同的'
            )
        except Exception as e:
            messagebox.showerror('错误', f'无法加载表情:\n{e}')

    def clear_emoji(self):
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

    # ──────────────── 滑块 ────────────────

    def _on_scale_change(self, value):
        if self.selected_idx is not None:
            face = self.faces_data[self.selected_idx]
            face['scale'] = float(value)
            self.scale_val_label.config(text=f'{float(value):.2f}×')
            self._rebuild_and_redraw()
            self._update_controls()

    def _sync_slider_to_selected(self):
        if self.selected_idx is not None:
            face = self.faces_data[self.selected_idx]
            self.scale_var.set(face['scale'])
            self.scale_val_label.config(text=f'{face["scale"]:.2f}×')

    # ──────────────── UI 状态更新 ────────────────

    def _update_controls(self):
        if self.selected_idx is not None:
            face = self.faces_data[self.selected_idx]
            has_emoji = face['emoji_img'] is not None
            ox, oy = face['offset']
            self.selected_label.config(
                text=f'当前选中: 人脸 #{self.selected_idx + 1}' +
                     (' 😀' if has_emoji else '')
            )
            self.offset_label.config(text=f'偏移: ({ox}, {oy})')
        else:
            self.selected_label.config(text='当前选中: (无)')
            self.offset_label.config(text='偏移: (0, 0)')

    # ──────────────── 保存 ────────────────

    def save_image(self):
        if not hasattr(self, '_composite') or self._composite is None:
            messagebox.showinfo('提示', '没有可保存的内容')
            return

        # 检查是否至少分配了一个表情
        has_any = any(f.get('emoji_img') is not None for f in self.faces_data)
        if not has_any:
            if not messagebox.askyesno('确认', '还没有加载任何表情，确定要保存吗？'):
                return

        path = filedialog.asksaveasfilename(
            title='保存图片',
            defaultextension='.png',
            filetypes=[
                ('PNG 图片', '*.png'),
                ('JPEG 图片', '*.jpg'),
                ('所有文件', '*.*')
            ]
        )
        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()
            save_img = self._composite
            if ext in ('.jpg', '.jpeg'):
                save_img = save_img.convert('RGB')
            save_img.save(path)
            self.status_var.set(f'已保存: {os.path.basename(path)}')
            messagebox.showinfo('完成', f'图片已保存到:\n{path}')
        except Exception as e:
            messagebox.showerror('错误', f'保存失败:\n{e}')


# ──────────────────────────────────────────────

if __name__ == '__main__':
    root = tk.Tk()
    app = FaceEmojiApp(root)
    root.mainloop()
