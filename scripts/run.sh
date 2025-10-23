#!/bin/bash

echo "========================================"
echo "   Hệ thống chấm điểm karaoke bằng AI"
echo "========================================"
echo

echo "Đang kiểm tra Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 chưa được cài đặt!"
    echo "Vui lòng cài đặt Python 3.8+ từ https://python.org"
    exit 1
fi

echo "✅ Python3 đã được cài đặt"
echo

echo "Đang cài đặt dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Lỗi khi cài đặt dependencies!"
    exit 1
fi

echo "✅ Dependencies đã được cài đặt"
echo

echo "Đang khởi động hệ thống..."
python3 main.py

