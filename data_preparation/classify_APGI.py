#!/usr/bin/env python3
"""
apgi2label.py
-------------
读取 APGI + mask → 直接输出净化后的 label 栅格：
    pkb_label/{region}_label.tif
依赖：rasterio  numpy  pathlib
"""

from pathlib import Path
import rasterio
from rasterio.features import sieve
import numpy as np

def make_labels(apgi_dir: Path,
                mask_dir: Path,
                label_dir: Path,
                thresholds: dict[str, float],
                sieve_size: int = 32,
                connectivity: int = 8):
    """
    Parameters
    ----------
    apgi_dir : Path            # {region}_APGI.tif
    mask_dir : Path            # {region}_mask.tif
    label_dir: Path            # 输出 pkb_label 目录
    thresholds : dict[str,float]
    sieve_size : int
        sieve 滤除面积阈值（像元数）
    connectivity : {4,8}
        sieve 连通方式
    """
    apgi_dir  = Path(apgi_dir)
    mask_dir  = Path(mask_dir)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    for region, th in thresholds.items():
        apgi_tif = apgi_dir / f"{region}_APGI.tif"
        msk_tif  = mask_dir / f"{region}_mask.tif"
        if not apgi_tif.exists():
            print(f"⚠️  缺少 {apgi_tif.name}，跳过")
            continue
        if not msk_tif.exists():
            print(f"⚠️  缺少 {msk_tif.name}，跳过")
            continue

        # ---------- 读数据 ----------
        with rasterio.open(apgi_tif) as apgi_src, rasterio.open(msk_tif) as msk_src:
            apgi = apgi_src.read(1, masked=True).filled(np.nan)
            msk  = msk_src.read(1, masked=True).filled(0)

            if apgi.shape != msk.shape:
                raise ValueError(f"{region}: APGI 与 mask 分辨率/尺寸不一致")

            # ---------- 阈值分类 ----------
            cls = (apgi >= th).astype(np.uint8)

            # ---------- 掩膜 ----------
            cls_masked = np.where(msk == 1, cls, 0).astype(np.uint8)

            # ---------- Sieve filter ----------
            cls_sf = sieve(cls_masked, size=sieve_size,
                           connectivity=connectivity).astype(np.uint8)

            meta = apgi_src.meta.copy()

        # ---------- 写最终 label ----------
        meta.update(dtype="uint8", nodata=0, compress="deflate")
        out_tif = label_dir / f"{region}_label.tif"
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(cls_sf, 1)

        print(f"✅  生成 {out_tif.name}  (阈值={th}, sieve={sieve_size})")

# ---------------- main ----------------
if __name__ == "__main__":
    root = Path(__file__).parents[1] / "Data"
    apgi_dir  = root / "APGI"
    mask_dir  = root / "Mask"
    label_dir = root / "pkb_label"

    thresholds = {
        "A": 0.35,
        "B": 0.25,
        "D": 0.30,
        "E": 0.22,
        "F": 0.25,
    }

    make_labels(apgi_dir, mask_dir, label_dir,
                thresholds,
                sieve_size=32,   # 可按需修改
                connectivity=8)
