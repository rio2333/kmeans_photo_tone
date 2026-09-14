#!/usr/bin/env python3
"""KMeans 照片主色调分类和两级编号重命名工具。"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_rgb(path: Path) -> np.ndarray | None:
    """使用 np.fromfile，支持 Windows 中文路径。"""
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except (OSError, cv2.error):
        return None


def main_color(path: Path, blocks: int, sample_limit: int) -> np.ndarray | None:
    image = read_rgb(path)
    if image is None:
        return None
    h, w = image.shape[:2]
    ratio = min(1.0, (sample_limit / max(h * w, 1)) ** 0.5)
    if ratio < 1:
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    pixels = image.reshape(-1, 3).astype(np.float32)
    if not len(pixels):
        return None
    k = min(blocks, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return centers[np.bincount(labels.ravel(), minlength=k).argmax()]


def collect(directory: Path, recursive: bool) -> list[Path]:
    paths = directory.rglob("*") if recursive else directory.glob("*")
    return sorted((p for p in paths if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: str(p).lower())


def mean_silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    """计算平均轮廓系数；只用于自动选择 K，不依赖 scikit-learn。"""
    distances = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
    scores = []
    for index, label in enumerate(labels):
        own = labels == label
        own_count = int(own.sum())
        if own_count < 2:  # 单成员类不能代表可靠色调类别
            scores.append(0.0)
            continue
        a = distances[index, own].sum() / (own_count - 1)
        b = min(distances[index, labels == other].mean() for other in np.unique(labels) if other != label)
        scores.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(scores))


def choose_group_count(colours: list[np.ndarray]) -> int:
    """在 1~10 个候选父类中自动选择色调分组数。"""
    data = np.asarray(colours, dtype=np.float32)
    if len(data) < 2 or np.max(np.ptp(data, axis=0)) < 8:
        return 1
    # 轮廓系数需两两距离；用最多 500 张均匀抽样的照片评估，适合大目录。
    sample = data[np.linspace(0, len(data) - 1, min(len(data), 500), dtype=int)]
    upper = min(10, len(sample) - 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    best_k, best_score = 2, -float("inf")
    for k in range(2, upper + 1):
        cv2.setRNGSeed(42 + k)
        _, labels, _ = cv2.kmeans(sample, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        score = mean_silhouette(sample, labels.ravel())
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def group_colours(colours: list[np.ndarray], group_count: int) -> np.ndarray:
    data = np.asarray(colours, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, group_count, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # 类别按 HSV 色相排序，让父类编号稳定且有色彩顺序。
    hsv = cv2.cvtColor(np.clip(centers, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    ordered = sorted(range(group_count), key=lambda i: (int(hsv[i, 0]), int(hsv[i, 1]) < 20, int(hsv[i, 2])))
    remap = {old: new + 1 for new, old in enumerate(ordered)}
    return np.asarray([remap[int(x)] for x in labels.ravel()])


def run() -> None:
    parser = argparse.ArgumentParser(description="按照片主色调分组并重命名为 01-001.jpg 格式")
    parser.add_argument("directory", nargs="?", type=Path, help="图片目录；不填则运行后询问")
    parser.add_argument("--groups", "-g", type=int, help="手动覆盖自动计算的父类数量（通常不需要）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不改名")
    parser.add_argument("--recursive", action="store_true", help="包含子目录")
    parser.add_argument("--color-clusters", type=int, default=4, help="每张图寻找主色的色块数（默认 4）")
    parser.add_argument("--sample-limit", type=int, default=10000, help="每张图最大采样像素数（默认 10000）")
    args = parser.parse_args()
    if args.directory is None:
        args.directory = Path(input("请输入图片文件夹的完整路径：").strip().strip('"'))
    if not args.directory.is_dir():
        parser.error(f"目录不存在：{args.directory}")
    if (args.groups is not None and args.groups < 1) or args.color_clusters < 1 or args.sample_limit < 1:
        parser.error("所有数量参数必须为正整数。")

    files, colours = [], []
    for path in collect(args.directory, args.recursive):
        colour = main_color(path, args.color_clusters, args.sample_limit)
        if colour is None:
            print(f"[跳过] 无法读取：{path.name}")
        else:
            files.append(path); colours.append(colour)
    if not files:
        parser.error("目录中没有可读取的图片。")
    if args.groups is None:
        args.groups = choose_group_count(colours)
        print(f"自动分析完成：建议使用 {args.groups} 个色调父类。")
    if args.groups > len(files):
        parser.error(f"父类数量不能超过照片数量（当前 {len(files)} 张）。")

    np.random.seed(42)
    parents = group_colours(colours, args.groups)
    parent_width, child_width = max(2, len(str(args.groups))), max(3, len(str(len(files))))
    counts: dict[int, int] = {}
    plans: list[tuple[Path, Path]] = []
    for source, parent in sorted(zip(files, parents), key=lambda x: (x[1], x[0].name.lower())):
        counts[parent] = counts.get(parent, 0) + 1
        target = source.with_name(f"{parent:0{parent_width}d}-{counts[parent]:0{child_width}d}{source.suffix.lower()}")
        plans.append((source, target))
    sources = {source for source, _ in plans}
    conflicts = [target for _, target in plans if target.exists() and target not in sources]
    if conflicts:
        parser.error(f"目标文件已存在，已停止以免覆盖：{conflicts[0]}")

    print(f"找到 {len(files)} 张图片，分为 {args.groups} 个父类：")
    for parent in range(1, args.groups + 1): print(f"  {parent:0{parent_width}d} 类：{counts.get(parent, 0)} 张")
    for source, target in plans: print(f"{'[预览] ' if args.dry_run else ''}{source.name} -> {target.name}")
    if args.dry_run:
        print("预览完成：移除 --dry-run 后才会真的改名。")
        return
    # 先移至唯一临时名称，再改为最终名称，防止彼此占用文件名。
    staged = []
    for index, (source, target) in enumerate(plans):
        temp = source.with_name(f".__tone_tmp_{index:06d}{source.suffix}")
        source.rename(temp); staged.append((temp, target))
    for temp, target in staged: temp.rename(target)
    print("重命名完成。")


if __name__ == "__main__":
    run()
