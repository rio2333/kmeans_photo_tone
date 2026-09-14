# 照片主色调分析程序

最方便的方式是双击 `启动照片主色调分类.bat`。窗口只会要求输入图片文件夹的完整路径；程序会自行从照片主色调中判断父类数量（在 1 到 10 类之间），不需要手动输入。

也可以在 PowerShell 中运行主程序后，直接按提示粘贴图片文件夹的完整路径：

```powershell
python .\photo_tone_classifier.py
```

建议先用预览模式，不会改变任何图片文件：

```powershell
python .\photo_tone_classifier.py "D:\照片" --dry-run
```

确认预览正确后，去掉 `--dry-run` 执行重命名。名称格式为 `父类-子类.扩展名`，例如 `01-001.jpg`、`01-002.jpg`、`02-001.png`。

程序已检查可用的 NumPy 与 OpenCV，并使用 OpenCV 内置 KMeans；不需要 scikit-learn。
