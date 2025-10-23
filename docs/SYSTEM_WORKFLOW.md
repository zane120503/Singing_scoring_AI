# Hệ thống chấm điểm karaoke - Workflow chi tiết

## 🎯 **Tổng quan hệ thống**

Hệ thống chấm điểm karaoke hoạt động theo 4 bước chính:

1. **Tách giọng hát** từ file karaoke
2. **Phát hiện phím âm nhạc** của beat và giọng hát
3. **So sánh phím âm nhạc** để đánh giá độ chính xác
4. **Tính điểm tổng thể** dựa trên nhiều tiêu chí

---

## 📋 **Chi tiết các bước**

### **BƯỚC 1: TÁCH GIỌNG HÁT** 🎤

**Mục tiêu**: Tách giọng hát khỏi beat nhạc trong file karaoke

**Công nghệ sử dụng**:
- **AI Audio Separator** (MDX-Net models)
- **Fallback**: Librosa HPSS (Harmonic-Percussive Separation)

**Quy trình**:
1. Load file karaoke (MP3/WAV)
2. Khởi tạo AI Audio Separator với MDX models
3. Tách giọng hát bằng AI model
4. Lưu file vocals đã tách vào thư mục output
5. Nếu AI thất bại → sử dụng fallback method

**Output**: File vocals riêng biệt (.mp3)

---

### **BƯỚC 2: PHÁT HIỆN PHÍM ÂM NHẠC** 🎹

**Mục tiêu**: Xác định phím âm nhạc của beat và giọng hát

**Công nghệ sử dụng**:
- **Docker Essentia AI** (cho beat)
- **Hybrid Key Detector** (cho vocals)
- **Traditional Librosa + Krumhansl profiles**

**Quy trình cho Beat**:
1. Load file beat nhạc
2. Preprocessing beat-specific
3. Sử dụng Docker Essentia AI
4. Fallback: Traditional chroma analysis

**Quy trình cho Vocals**:
1. Load file vocals đã tách
2. Preprocessing vocals-specific (harmonic separation)
3. Hybrid detection với multiple methods:
   - Docker Essentia AI (trọng số thấp)
   - Traditional Librosa (trọng số cao)
   - Vocals-specific analysis (trọng số cao)
   - Enhanced Chroma (trọng số cao)
4. Weighted voting để chọn kết quả tốt nhất

**Output**: 
- Beat key: `{"key": "C", "scale": "major", "confidence": 0.82}`
- Vocals key: `{"key": "A", "scale": "minor", "confidence": 0.75}`

---

### **BƯỚC 3: SO SÁNH PHÍM ÂM NHẠC** 🎵

**Mục tiêu**: Đánh giá độ chính xác về phím âm nhạc

**Phương pháp**:
1. **Exact match**: Cùng key và scale (100 điểm)
2. **Relative keys**: Key tương đối (80 điểm)
3. **Parallel keys**: Key song song (60 điểm)
4. **Circle of fifths**: Dựa trên khoảng cách (40-80 điểm)
5. **No match**: Không liên quan (0 điểm)

**Ví dụ**:
- Beat: C major, Vocals: A minor → Relative keys → 80 điểm
- Beat: C major, Vocals: C major → Exact match → 100 điểm

---

### **BƯỚC 4: TÍNH ĐIỂM TỔNG THỂ** 📊

**Mục tiêu**: Tính điểm tổng thể dựa trên 5 tiêu chí

**Các tiêu chí và trọng số**:
1. **Key Accuracy** (30%): Độ chính xác về phím âm nhạc
2. **Pitch Accuracy** (25%): Độ chính xác về cao độ
3. **Rhythm Accuracy** (20%): Độ chính xác về nhịp điệu
4. **Timing Accuracy** (15%): Độ chính xác về thời gian
5. **Vocal Quality** (10%): Chất lượng giọng hát

**Công thức tính điểm**:
```
Overall Score = Σ(Score_i × Weight_i)
```

**Chi tiết từng tiêu chí**:

#### **Key Accuracy (30%)**:
- Sử dụng kết quả từ bước 3
- Score = Key similarity score từ KeyDetector

#### **Pitch Accuracy (25%)**:
- Phân tích fundamental frequency (F0)
- So sánh pitch contour giữa vocals và beat
- Sử dụng YIN algorithm để extract F0

#### **Rhythm Accuracy (20%)**:
- Phân tích tempo và beat tracking
- So sánh rhythmic patterns
- Sử dụng librosa beat tracking

#### **Timing Accuracy (15%)**:
- Phân tích onset detection
- So sánh timing giữa vocals và beat
- Đánh giá độ đồng bộ

#### **Vocal Quality (10%)**:
- Phân tích spectral features
- Đánh giá chất lượng âm thanh
- Noise ratio, harmonic content

---

## 🔄 **Workflow tổng thể**

```
Input: File Karaoke (Beat + Vocals)
    ↓
[1] AI Audio Separator → Vocals File
    ↓
[2a] Beat Key Detection → Beat Key
[2b] Vocals Key Detection → Vocals Key
    ↓
[3] Key Comparison → Key Accuracy Score
    ↓
[4] Overall Scoring → Final Score
    ↓
Output: Detailed Results + Feedback
```

---

## 📁 **Cấu trúc file và thư mục**

```
singing scoring AI/
├── src/
│   ├── ai/
│   │   ├── advanced_audio_processor.py    # AI Audio Separator
│   │   ├── advanced_key_detector.py       # Key Detection
│   │   ├── voice_activity_detector.py     # Voice Activity Detection
│   │   └── audio_slicer.py               # Audio Slicing
│   ├── core/
│   │   └── scoring_system.py             # Scoring Logic
│   └── gui/
│       └── gui.py                        # GUI Interface
├── assets/
│   └── models/
│       └── mdx_models/                   # AI Models
├── output/                               # Output Files
└── main.py                              # Entry Point
```

---

## 🎯 **Các tính năng nâng cao**

### **Smart Audio Processing**:
- Voice Activity Detection để tìm đoạn có giọng hát
- Audio Slicing để cắt đoạn 20s thay vì xử lý toàn bộ file
- Tối ưu hóa thời gian xử lý

### **Hybrid Key Detection**:
- Kết hợp multiple methods cho độ chính xác cao
- Weighted voting system
- Vocals-specific preprocessing

### **Real-time Processing**:
- GUI với progress tracking
- Threading để không block interface
- Error handling và fallback methods

---

## 📊 **Kết quả đầu ra**

```json
{
  "overall_score": 75.5,
  "detailed_scores": {
    "key_accuracy": 80.0,
    "pitch_accuracy": 70.0,
    "rhythm_accuracy": 75.0,
    "timing_accuracy": 80.0,
    "vocal_quality": 85.0
  },
  "key_info": {
    "beat_key": {"key": "C", "scale": "major", "confidence": 0.82},
    "vocals_key": {"key": "A", "scale": "minor", "confidence": 0.75},
    "key_similarity": 80
  },
  "feedback": "Độ chính xác phím: Khá tốt\nĐộ chính xác cao độ: Cần luyện tập nhiều hơn"
}
```

---

## 🚀 **Tối ưu hóa và cải tiến**

1. **Giảm thời gian xử lý**: Smart Audio Processing chỉ xử lý đoạn 20s
2. **Tăng độ chính xác**: Hybrid Key Detection với multiple methods
3. **Cải thiện UX**: Real-time GUI với progress tracking
4. **Robust error handling**: Fallback methods khi AI thất bại
