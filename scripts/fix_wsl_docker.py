#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script khắc phục vấn đề WSL với Docker Desktop
"""

import subprocess
import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_as_admin():
    """Kiểm tra xem script có chạy với quyền admin không"""
    import ctypes
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def fix_wsl_docker():
    """Khắc phục vấn đề WSL với Docker"""
    logger.info("="*60)
    logger.info("🔧 KHẮC PHỤC VẤN ĐỀ WSL VỚI DOCKER DESKTOP")
    logger.info("="*60)
    
    # Kiểm tra quyền admin
    if not run_as_admin():
        logger.warning("⚠️ Script cần quyền Administrator")
        logger.info("   💡 Hãy chạy lại với quyền Administrator:")
        logger.info("      Right-click PowerShell → Run as Administrator")
        logger.info("      Sau đó chạy: python scripts/fix_wsl_docker.py")
        return False
    
    logger.info("\n1️⃣ Đóng Docker Desktop...")
    try:
        # Tắt Docker Desktop
        subprocess.run("taskkill /F /IM Docker Desktop.exe", shell=True, capture_output=True)
        logger.info("   ✅ Đã gửi lệnh đóng Docker Desktop")
        logger.info("   ⏳ Đợi 3 giây để Docker Desktop đóng hoàn toàn...")
        time.sleep(3)
    except Exception as e:
        logger.warning(f"   ⚠️ Không thể đóng Docker Desktop tự động: {e}")
        logger.info("   💡 Hãy đóng Docker Desktop thủ công:")
        logger.info("      - Right-click biểu tượng Docker ở system tray")
        logger.info("      - Chọn 'Quit Docker Desktop'")
        input("\n   Nhấn Enter sau khi đã đóng Docker Desktop...")
    
    logger.info("\n2️⃣ Shutdown WSL...")
    try:
        result = subprocess.run("wsl --shutdown", shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("   ✅ WSL đã được shutdown")
        else:
            logger.warning(f"   ⚠️ Lệnh shutdown WSL: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.warning("   ⚠️ Lệnh shutdown WSL timeout (có thể WSL đã shutdown)")
    except Exception as e:
        logger.error(f"   ❌ Lỗi shutdown WSL: {e}")
        return False
    
    logger.info("   ⏳ Đợi 2 giây...")
    time.sleep(2)
    
    logger.info("\n3️⃣ Kiểm tra WSL...")
    try:
        result = subprocess.run("wsl --list --verbose", shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("   ✅ WSL đang hoạt động bình thường")
            if result.stdout:
                logger.info(f"   📊 WSL distributions:\n{result.stdout}")
        else:
            logger.warning("   ⚠️ WSL có vấn đề")
            logger.warning(f"   Error: {result.stderr}")
    except Exception as e:
        logger.warning(f"   ⚠️ Không thể kiểm tra WSL: {e}")
    
    logger.info("\n4️⃣ Hướng dẫn khởi động lại Docker Desktop...")
    logger.info("   ✅ Đã hoàn thành các bước khắc phục!")
    logger.info("\n   💡 BƯỚC TIẾP THEO:")
    logger.info("      1. Khởi động Docker Desktop từ Start Menu")
    logger.info("      2. Đợi Docker Desktop khởi động hoàn toàn")
    logger.info("      3. Kiểm tra Docker hoạt động bằng lệnh: docker ps")
    logger.info("      4. Nếu vẫn lỗi, thử restart máy tính")
    logger.info("\n" + "="*60)
    return True

if __name__ == "__main__":
    success = fix_wsl_docker()
    sys.exit(0 if success else 1)


