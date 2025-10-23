import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import logging
from src.ai.advanced_audio_processor import AdvancedAudioProcessor
from src.ai.advanced_key_detector import AdvancedKeyDetector  # Sử dụng AdvancedKeyDetector với Essentia
from src.ai.optimized_audio_processor import OptimizedAudioProcessor  # Workflow tối ưu hóa
from src.core.scoring_system import KaraokeScoringSystem
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaraokeScoringGUI:
    """Giao diện GUI cho hệ thống chấm điểm karaoke"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống chấm điểm giọng hát karaoke bằng AI")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Khởi tạo các components
        logger.info("🔄 Đang khởi tạo các components...")
        self.audio_processor = AdvancedAudioProcessor(fast_mode=False)  # Sử dụng AI Audio Separator
        self.key_detector = AdvancedKeyDetector()  # Sử dụng AdvancedKeyDetector với Essentia AI
        self.scoring_system = KaraokeScoringSystem()
        
        # Khởi tạo Optimized Audio Processor cho workflow mới
        self.optimized_processor = OptimizedAudioProcessor()
        self.use_optimized_workflow = True  # Sử dụng workflow tối ưu hóa mặc định
        
        logger.info("✅ Tất cả components đã được khởi tạo!")
        
        # Biến lưu trữ file paths
        self.karaoke_file = None
        self.beat_file = None
        self.vocals_file = None
        
        # Tạo giao diện
        self.create_widgets()
        
        # Thêm checkbox cho workflow selection
        self.create_workflow_selection()
        
    def create_widgets(self):
        """Tạo các widget cho giao diện"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="🎤 Hệ thống chấm điểm giọng hát karaoke bằng AI 🎵",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - File selection
        left_panel = tk.Frame(main_frame, bg='#ffffff', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # File selection section
        file_frame = tk.LabelFrame(left_panel, text="📁 Chọn file âm thanh", font=("Arial", 12, "bold"), bg='#ffffff')
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Karaoke file selection
        karaoke_frame = tk.Frame(file_frame, bg='#ffffff')
        karaoke_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(karaoke_frame, text="File karaoke:", font=("Arial", 10), bg='#ffffff').pack(anchor=tk.W)
        karaoke_btn_frame = tk.Frame(karaoke_frame, bg='#ffffff')
        karaoke_btn_frame.pack(fill=tk.X, pady=2)
        
        self.karaoke_label = tk.Label(karaoke_btn_frame, text="Chưa chọn file", fg='#7f8c8d', bg='#ffffff')
        self.karaoke_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Button(
            karaoke_btn_frame, 
            text="Chọn file", 
            command=self.select_karaoke_file,
            bg='#3498db',
            fg='white',
            font=("Arial", 9)
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Beat file selection
        beat_frame = tk.Frame(file_frame, bg='#ffffff')
        beat_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(beat_frame, text="File beat nhạc:", font=("Arial", 10), bg='#ffffff').pack(anchor=tk.W)
        beat_btn_frame = tk.Frame(beat_frame, bg='#ffffff')
        beat_btn_frame.pack(fill=tk.X, pady=2)
        
        self.beat_label = tk.Label(beat_btn_frame, text="Chưa chọn file", fg='#7f8c8d', bg='#ffffff')
        self.beat_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Button(
            beat_btn_frame, 
            text="Chọn file", 
            command=self.select_beat_file,
            bg='#3498db',
            fg='white',
            font=("Arial", 9)
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Process button
        process_frame = tk.Frame(left_panel, bg='#ffffff')
        process_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.process_btn = tk.Button(
            process_frame,
            text="🚀 Bắt đầu phân tích và chấm điểm",
            command=self.start_analysis,
            bg='#e74c3c',
            fg='white',
            font=("Arial", 12, "bold"),
            height=2
        )
        self.process_btn.pack(fill=tk.X)
        
        # Progress bar
        self.progress = ttk.Progressbar(process_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_label = tk.Label(
            process_frame, 
            text="Sẵn sàng phân tích", 
            font=("Arial", 10),
            bg='#ffffff',
            fg='#27ae60'
        )
        self.status_label.pack(pady=5)
        
        # AI Model Status
        ai_status_frame = tk.Frame(left_panel, bg='#ffffff')
        ai_status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(ai_status_frame, text="🤖 Trạng thái AI Models:", font=("Arial", 10, "bold"), bg='#ffffff').pack(anchor=tk.W)
        
        self.ai_status_text = scrolledtext.ScrolledText(
            ai_status_frame, 
            height=4, 
            font=("Arial", 9),
            bg='#f8f9fa',
            wrap=tk.WORD
        )
        self.ai_status_text.pack(fill=tk.X, pady=2)
        
        # Cập nhật trạng thái AI models
        self.update_ai_status()
        
        # Right panel - Results
        right_panel = tk.Frame(main_frame, bg='#ffffff', relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Results section
        results_frame = tk.LabelFrame(right_panel, text="📊 Kết quả chấm điểm", font=("Arial", 12, "bold"), bg='#ffffff')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Overall score display
        score_frame = tk.Frame(results_frame, bg='#ffffff')
        score_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.overall_score_label = tk.Label(
            score_frame,
            text="Điểm tổng thể: --",
            font=("Arial", 14, "bold"),
            bg='#ffffff',
            fg='#2c3e50'
        )
        self.overall_score_label.pack()
        
        self.grade_label = tk.Label(
            score_frame,
            text="Xếp loại: --",
            font=("Arial", 12),
            bg='#ffffff',
            fg='#e74c3c'
        )
        self.grade_label.pack()
        
        # Detailed scores
        scores_frame = tk.Frame(results_frame, bg='#ffffff')
        scores_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(scores_frame, text="Chi tiết điểm số:", font=("Arial", 11, "bold"), bg='#ffffff').pack(anchor=tk.W)
        
        # Create treeview for detailed scores
        columns = ('Tiêu chí', 'Điểm', 'Trọng số')
        self.scores_tree = ttk.Treeview(scores_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.scores_tree.heading(col, text=col)
            self.scores_tree.column(col, width=100)
        
        self.scores_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Feedback section
        feedback_frame = tk.Frame(results_frame, bg='#ffffff')
        feedback_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(feedback_frame, text="Phản hồi:", font=("Arial", 11, "bold"), bg='#ffffff').pack(anchor=tk.W)
        
        self.feedback_text = scrolledtext.ScrolledText(
            feedback_frame, 
            height=6, 
            font=("Arial", 10),
            bg='#f8f9fa',
            wrap=tk.WORD
        )
        self.feedback_text.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_workflow_selection(self):
        """Tạo checkbox để chọn workflow"""
        # Workflow selection frame
        workflow_frame = tk.Frame(self.root, bg='#f0f0f0')
        workflow_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(workflow_frame, text="Workflow:", font=("Arial", 11, "bold"), bg='#f0f0f0').pack(side=tk.LEFT)
        
        # Checkbox cho optimized workflow
        self.optimized_workflow_var = tk.BooleanVar(value=True)
        self.optimized_checkbox = tk.Checkbutton(
            workflow_frame,
            text="🚀 Optimized Workflow",
            variable=self.optimized_workflow_var,
            command=self.on_workflow_change,
            bg='#f0f0f0',
            font=("Arial", 10)
        )
        self.optimized_checkbox.pack(side=tk.LEFT, padx=10)
        
        # Info label
        self.workflow_info_label = tk.Label(
            workflow_frame,
            text="✅ Optimized: Voice Detection → Audio Separation → Key Detection → Scoring",
            font=("Arial", 9),
            fg='#27ae60',
            bg='#f0f0f0'
        )
        self.workflow_info_label.pack(side=tk.LEFT, padx=10)
    
    def on_workflow_change(self):
        """Xử lý khi thay đổi workflow"""
        if self.optimized_workflow_var.get():
            self.workflow_info_label.config(
                text="✅ Optimized: Voice Detection → Audio Separation → Key Detection → Scoring",
                fg='#27ae60'
            )
        else:
            self.workflow_info_label.config(
                text="⚡ Standard: Full Audio Separation → Key Detection → Scoring",
                fg='#f39c12'
            )
        
    def select_karaoke_file(self):
        """Chọn file karaoke"""
        file_path = filedialog.askopenfilename(
            title="Chọn file karaoke",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a"), ("All files", "*.*")]
        )
        if file_path:
            self.karaoke_file = file_path
            filename = os.path.basename(file_path)
            self.karaoke_label.config(text=filename, fg='#27ae60')
    
    def select_beat_file(self):
        """Chọn file beat nhạc"""
        file_path = filedialog.askopenfilename(
            title="Chọn file beat nhạc",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a"), ("All files", "*.*")]
        )
        if file_path:
            self.beat_file = file_path
            filename = os.path.basename(file_path)
            self.beat_label.config(text=filename, fg='#27ae60')
    
    def update_ai_status(self):
        """Cập nhật trạng thái AI models"""
        try:
            status_text = ""
            
            # Kiểm tra Audio Processor AI model
            if hasattr(self.audio_processor, 'model') and self.audio_processor.model is not None:
                if self.audio_processor.model == "Audio_Separator_AI":
                    status_text += "✅ AI Audio Separator: Hoạt động (MDX-Net)\n"
                elif self.audio_processor.model == "Fast_Mode":
                    status_text += "⚡ AI Audio Separator: Fast Mode (Tốc độ cao)\n"
                else:
                    status_text += f"✅ AI Audio Separator: Hoạt động ({self.audio_processor.model})\n"
            else:
                status_text += "⚠️ AI Audio Separator: Fallback (Librosa)\n"
            
            # Kiểm tra Essentia AI model
            if hasattr(self.key_detector, 'essentia_available') and self.key_detector.essentia_available:
                status_text += "✅ Essentia AI Key Detector: Hoạt động\n"
            elif hasattr(self.key_detector, 'docker_available') and self.key_detector.docker_available:
                status_text += "🐳 Docker Essentia AI Key Detector: Hoạt động\n"
            else:
                status_text += "⚠️ Essentia AI Key Detector: Fallback (Traditional)\n"
            
            # Thông tin device
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            status_text += f"📱 Device: {device}\n"
            
            # Cập nhật text widget
            self.ai_status_text.delete(1.0, tk.END)
            self.ai_status_text.insert(1.0, status_text)
            
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật trạng thái AI: {e}")
            self.ai_status_text.delete(1.0, tk.END)
            self.ai_status_text.insert(1.0, f"❌ Lỗi: {str(e)}")
    
    def start_analysis(self):
        """Bắt đầu phân tích trong thread riêng"""
        if not self.karaoke_file or not self.beat_file:
            messagebox.showerror("Lỗi", "Vui lòng chọn cả file karaoke và file beat nhạc!")
            return
        
        # Disable button và start progress
        self.process_btn.config(state='disabled')
        self.progress.start()
        self.status_label.config(text="Đang phân tích...", fg='#f39c12')
        
        # Start analysis in separate thread
        thread = threading.Thread(target=self.analyze_audio)
        thread.daemon = True
        thread.start()
    
    def analyze_audio(self):
        """Phân tích âm thanh và chấm điểm"""
        try:
            logger.info("🎤 Bắt đầu phân tích âm thanh...")
            
            # Kiểm tra workflow được chọn
            if self.optimized_workflow_var.get():
                logger.info("🚀 Sử dụng Optimized Workflow...")
                self.analyze_audio_optimized()
            else:
                logger.info("⚡ Sử dụng Standard Workflow...")
                self.analyze_audio_standard()
            
        except Exception as e:
            error_msg = f"Lỗi trong quá trình phân tích: {str(e)}"
            self.root.after(0, lambda: self.show_error(error_msg))
        finally:
            # Re-enable button và stop progress
            self.root.after(0, self.reset_ui)
    
    def analyze_audio_optimized(self):
        """Phân tích âm thanh với workflow tối ưu hóa"""
        try:
            logger.info("🚀 Bắt đầu phân tích với Optimized Workflow...")
            
            # Sử dụng Optimized Audio Processor
            result = self.optimized_processor.process_karaoke_optimized(
                self.karaoke_file, 
                self.beat_file
            )
            
            if result["success"]:
                logger.info("✅ Optimized workflow hoàn thành!")
                
                # Cập nhật giao diện với kết quả
                self.root.after(0, lambda: self.display_optimized_results(result))
            else:
                error_msg = f"Optimized workflow thất bại: {result['error']}"
                self.root.after(0, lambda: self.show_error(error_msg))
                
        except Exception as e:
            logger.error(f"❌ Lỗi trong optimized workflow: {e}")
            raise e
    
    def analyze_audio_standard(self):
        """Phân tích âm thanh với workflow chuẩn"""
        try:
            logger.info("⚡ Bắt đầu phân tích với Standard Workflow...")
            
            # Step 1: Tách giọng hát
            self.update_status("Đang tách giọng hát...")
            logger.info("🔄 Step 1: Tách giọng hát bằng AI Audio Separator...")
            vocals_path = self.audio_processor.separate_vocals(self.karaoke_file)
            self.vocals_file = vocals_path
            logger.info(f"✅ Vocals file created: {vocals_path}")
            
            # Step 2: Phát hiện phím âm nhạc
            self.update_status("Đang phát hiện phím âm nhạc...")
            logger.info("🔄 Step 2: Phát hiện phím âm nhạc bằng AI Key Detector...")
            beat_key = self.key_detector.detect_key(self.beat_file, "beat")
            vocals_key = self.key_detector.detect_key(vocals_path, "vocals")
            logger.info(f"✅ Beat key: {beat_key['key']} {beat_key['scale']}")
            logger.info(f"✅ Vocals key: {vocals_key['key']} {vocals_key['scale']}")
            
            # Step 3: So sánh phím
            logger.info("🔄 Step 3: So sánh phím âm nhạc...")
            key_comparison = self.key_detector.compare_keys(beat_key, vocals_key)
            logger.info(f"✅ Key similarity score: {key_comparison['score']}/100")
            
            # Step 4: Tính điểm tổng thể
            self.update_status("Đang tính điểm...")
            logger.info("🔄 Step 4: Tính điểm tổng thể...")
            scoring_result = self.scoring_system.calculate_overall_score(
                self.karaoke_file, self.beat_file, vocals_path
            )
            logger.info(f"✅ Overall score: {scoring_result['overall_score']}/100")
            
            # Step 5: Cập nhật giao diện
            self.update_status("Hoàn thành!")
            logger.info("🎉 Phân tích hoàn thành!")
            self.root.after(0, lambda: self.display_results(scoring_result, beat_key, vocals_key, key_comparison))
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong standard workflow: {e}")
            raise e
    
    def update_status(self, message):
        """Cập nhật trạng thái"""
        self.root.after(0, lambda: self.status_label.config(text=message, fg='#f39c12'))
    
    def display_optimized_results(self, result):
        """Hiển thị kết quả từ optimized workflow"""
        try:
            # Cập nhật trạng thái
            self.update_status("Hoàn thành!")
            
            # Lấy thông tin từ result
            scoring_result = result["scoring"]
            beat_key = result["key_detection"]["beat_key"]
            vocals_key = result["key_detection"]["vocals_key"]
            key_comparison = result["key_detection"]["key_comparison"]
            
            # Hiển thị kết quả chính
            self.display_results(scoring_result, beat_key, vocals_key, key_comparison)
            
            # Thêm thông tin optimized workflow
            optimized_info = f"""
🚀 OPTIMIZED WORKFLOW INFO:
📁 Files processed:
   • Karaoke file: {os.path.basename(result['processed_files']['karaoke_file'])}
   • Vocals file: {os.path.basename(result['processed_files']['vocals_file'])}
🎤 Voice Detection:
   • Voice segments found: {len(result['voice_detection']['voice_segments'])}
   • Selected segment: {result['voice_detection']['selected_voice']['start']:.2f}s - {result['voice_detection']['selected_voice']['end']:.2f}s
            """
            
            # Thêm vào feedback
            current_feedback = self.feedback_text.get("1.0", tk.END)
            self.feedback_text.delete("1.0", tk.END)
            self.feedback_text.insert("1.0", current_feedback + optimized_info)
            
        except Exception as e:
            logger.error(f"❌ Lỗi hiển thị optimized results: {e}")
    
    def display_results(self, scoring_result, beat_key, vocals_key, key_comparison):
        """Hiển thị kết quả"""
        # Overall score
        overall_score = scoring_result['overall_score']
        grade = scoring_result['grade']
        
        self.overall_score_label.config(text=f"Điểm tổng thể: {overall_score}/100")
        self.grade_label.config(text=f"Xếp loại: {grade}")
        
        # Detailed scores
        for item in self.scores_tree.get_children():
            self.scores_tree.delete(item)
        
        criterion_names = {
            'key_accuracy': 'Độ chính xác phím',
            'pitch_accuracy': 'Độ chính xác cao độ',
            'rhythm_accuracy': 'Độ chính xác nhịp điệu',
            'timing_accuracy': 'Độ chính xác thời gian',
            'vocal_quality': 'Chất lượng giọng hát',
            'energy_consistency': 'Tính nhất quán năng lượng',
            'pronunciation': 'Phát âm'
        }
        
        for criterion, score in scoring_result['detailed_scores'].items():
            weight = scoring_result['weights'][criterion] * 100
            criterion_name = criterion_names.get(criterion, criterion)
            self.scores_tree.insert('', 'end', values=(criterion_name, f"{score:.1f}", f"{weight:.1f}%"))
        
        # Feedback
        self.feedback_text.delete(1.0, tk.END)
        
        # Key information
        feedback_text = f"🎵 Thông tin phím âm nhạc:\n"
        feedback_text += f"Beat nhạc: {beat_key['key']} {beat_key['scale']} (độ tin cậy: {beat_key['confidence']:.2f})\n"
        feedback_text += f"Giọng hát: {vocals_key['key']} {vocals_key['scale']} (độ tin cậy: {vocals_key['confidence']:.2f})\n"
        feedback_text += f"Điểm tương đồng phím: {key_comparison['score']}/100\n\n"
        
        # Detailed feedback
        feedback_text += "📝 Phản hồi chi tiết:\n"
        for feedback in scoring_result['feedback']:
            feedback_text += f"• {feedback}\n"
        
        self.feedback_text.insert(1.0, feedback_text)
    
    def show_error(self, error_msg):
        """Hiển thị lỗi"""
        messagebox.showerror("Lỗi", error_msg)
        self.status_label.config(text="Có lỗi xảy ra", fg='#e74c3c')
    
    def reset_ui(self):
        """Reset giao diện sau khi hoàn thành"""
        self.process_btn.config(state='normal')
        self.progress.stop()
        self.status_label.config(text="Sẵn sàng phân tích", fg='#27ae60')

def main():
    """Hàm main để chạy ứng dụng"""
    root = tk.Tk()
    app = KaraokeScoringGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
