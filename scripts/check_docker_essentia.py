#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script kiểm tra và khắc phục Docker Essentia
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_docker():
    """Kiểm tra Docker"""
    logger.info("="*60)
    logger.info("🔍 KIỂM TRA DOCKER ESSENTIA")
    logger.info("="*60)
    
    # Bước 1: Kiểm tra Docker có cài đặt không
    logger.info("\n1️⃣ Kiểm tra Docker installation...")
    try:
        result = subprocess.run("docker --version", shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info(f"   ✅ {result.stdout.strip()}")
        else:
            logger.error(f"   ❌ Docker không được cài đặt")
            logger.error(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"   ❌ Lỗi kiểm tra Docker: {e}")
        return False
    
    # Bước 2: Kiểm tra Docker daemon
    logger.info("\n2️⃣ Kiểm tra Docker daemon...")
    try:
        result = subprocess.run("docker info", shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("   ✅ Docker daemon đang chạy")
        else:
            logger.error("   ❌ Docker daemon không chạy hoặc không thể kết nối")
            logger.error(f"   Error: {result.stderr}")
            logger.info("\n   💡 GIẢI PHÁP:")
            logger.info("      1. Mở Docker Desktop")
            logger.info("      2. Đợi Docker Desktop khởi động hoàn toàn")
            logger.info("      3. Chạy lại script này")
            return False
    except subprocess.TimeoutExpired:
        logger.error("   ❌ Docker command timeout - Docker không phản hồi")
        logger.info("\n   💡 Hãy khởi động lại Docker Desktop")
        return False
    except Exception as e:
        logger.error(f"   ❌ Lỗi kiểm tra Docker daemon: {e}")
        return False
    
    # Bước 3: Kiểm tra container essentia-karaoke
    logger.info("\n3️⃣ Kiểm tra container essentia-karaoke...")
    try:
        result = subprocess.run('docker ps -a --filter name=essentia-karaoke --format "{{.Names}}|{{.Status}}"', 
                              shell=True, capture_output=True, text=True, timeout=5)
        
        if result.stdout.strip():
            parts = result.stdout.strip().split('|')
            container_name = parts[0]
            container_status = parts[1] if len(parts) > 1 else "unknown"
            
            logger.info(f"   ✅ Tìm thấy container: {container_name}")
            logger.info(f"   📊 Status: {container_status}")
            
            # Kiểm tra container có đang chạy không
            if "Up" in container_status:
                logger.info("   ✅ Container đang chạy")
            else:
                logger.warning("   ⚠️ Container không đang chạy")
                logger.info("\n   💡 Khởi động container:")
                logger.info("      docker start essentia-karaoke")
                
                # Thử khởi động
                start_result = subprocess.run("docker start essentia-karaoke", 
                                            shell=True, capture_output=True, text=True, timeout=10)
                if start_result.returncode == 0:
                    logger.info("   ✅ Đã khởi động container!")
                else:
                    logger.error(f"   ❌ Không thể khởi động container: {start_result.stderr}")
                    return False
        else:
            logger.warning("   ⚠️ Container 'essentia-karaoke' chưa được tạo")
            logger.info("\n   💡 Tạo container:")
            logger.info("      docker run -d --name essentia-karaoke mtgupf/essentia:latest")
            
            response = input("\n   ❓ Bạn có muốn tạo container ngay bây giờ? (y/n): ")
            if response.lower() == 'y':
                logger.info("   🔄 Đang tạo container...")
                create_result = subprocess.run("docker run -d --name essentia-karaoke mtgupf/essentia:latest", 
                                             shell=True, capture_output=True, text=True, timeout=60)
                if create_result.returncode == 0:
                    logger.info("   ✅ Container đã được tạo!")
                else:
                    logger.error(f"   ❌ Lỗi tạo container: {create_result.stderr}")
                    return False
            else:
                return False
                
    except Exception as e:
        logger.error(f"   ❌ Lỗi kiểm tra container: {e}")
        return False
    
    # Bước 4: Test Essentia trong container
    logger.info("\n4️⃣ Kiểm tra Essentia trong container...")
    try:
        test_cmd = 'docker exec essentia-karaoke python3 -c "import essentia.standard as es; print(\'OK\')"'
        result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "OK" in result.stdout:
            logger.info("   ✅ Essentia hoạt động bình thường trong container!")
            logger.info("\n" + "="*60)
            logger.info("🎉 DOCKER ESSENTIA SẴN SÀNG!")
            logger.info("="*60)
            return True
        else:
            logger.error("   ❌ Essentia không khả dụng trong container")
            logger.error(f"   Return code: {result.returncode}")
            logger.error(f"   stdout: {result.stdout}")
            logger.error(f"   stderr: {result.stderr}")
            logger.info("\n   💡 Thử cài lại container:")
            logger.info("      docker rm -f essentia-karaoke")
            logger.info("      docker run -d --name essentia-karaoke mtgupf/essentia:latest")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("   ❌ Command timeout - Container có thể không phản hồi")
        return False
    except Exception as e:
        logger.error(f"   ❌ Lỗi test Essentia: {e}")
        return False

if __name__ == "__main__":
    success = check_docker()
    sys.exit(0 if success else 1)


