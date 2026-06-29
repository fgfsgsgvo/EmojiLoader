# EmojiLoader - 人脸表情包工具

给图片中的人脸自动覆盖自定义表情包。支持多人脸、每人不同表情、自动旋转对齐、大小调节和拖拽微调。

## 使用方法

### 方式一：Python 脚本运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行
python face_emoji.py
```

### 方式二：双击 exe（无需 Python）

```bash
# 双击打包
build_exe.bat
# 生成的 exe 在 dist/EmojiLoader.exe
```

## 操作流程

```
1. [📷 打开图片]         → 选择一张含有人脸的照片
2. [👤 检测人脸]         → 自动检测并框出所有人脸
3. 点击选中某个人脸       → 蓝色高亮表示已选中
4. [😀 给此人加载表情]   → 选择一张表情图（PNG/JPG）
5. 拖动「大小」滑块       → 调节该人脸的表情大小
6. 拖拽表情               → 微调位置
7. 重复 3-6 为每个人分配
8. [💾 保存图片]         → 保存合成结果
```

## 小技巧

- **Tab 键** — 快速切换到下一个人脸
- **方向键 ← → ↑ ↓** — 微调表情位置（每次 2px）
- **ESC** — 取消选中
- **点击画布上的人脸** — 直接选中
- 表情图建议用 **PNG 透明背景**，效果最好
- 人脸倾斜时表情会自动旋转对齐

## 文件结构

```
EmojiLoader/
├── face_emoji.py         # 主程序
├── requirements.txt      # 依赖列表
├── build_exe.bat         # 打包脚本
├── README.md             # 本文件
└── dist/                 # 打包输出（运行 build_exe.bat 后生成）
```

## 技术栈

- Python + Tkinter（GUI）
- OpenCV Haar Cascade（人脸检测 + 眼睛检测）
- Pillow（图像合成）
- PyInstaller（打包 exe）
