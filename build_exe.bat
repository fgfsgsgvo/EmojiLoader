@echo off
REM EmojiLoader 一键打包脚本
REM 用法: 双击运行或在命令行执行 build_exe.bat

echo ========================================
echo   EmojiLoader - 打包中...
echo ========================================

REM 安装依赖（如果还没装）
echo [1/3] 检查依赖...
pip install opencv-python pyinstaller -q

REM 清理旧的打包
if exist dist rmdir /s /q dist

REM 打包
echo [2/3] 打包中...
pyinstaller --onefile --windowed --name "EmojiLoader" --clean face_emoji.py

REM 检查结果
if exist dist\EmojiLoader.exe (
    echo [3/3] 打包成功!
    echo.
    echo 输出文件: %cd%\dist\EmojiLoader.exe
    echo.
    echo 可以直接双击运行啦!
) else (
    echo [3/3] 打包失败，请检查错误信息
)

pause
