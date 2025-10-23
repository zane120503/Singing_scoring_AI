
def _detect_voice_activity_improved(self, audio: np.ndarray, sr: int) -> List[Dict]:
    """Phát hiện voice activity cải thiện"""
    try:
        # 1. RMS Energy Analysis
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        # Adaptive threshold - thấp hơn để phát hiện voice nhẹ
        rms_threshold = np.percentile(rms, 20)  # Sử dụng percentile thấp hơn
        
        # 2. Spectral Centroid Analysis
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Voice frequency range (80-3000 Hz)
        voice_low = 80
        voice_high = 3000
        
        # 3. Zero Crossing Rate Analysis
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Voice có ZCR trung bình
        zcr_low = np.percentile(zcr, 10)
        zcr_high = np.percentile(zcr, 90)
        
        # 4. Spectral Rolloff Analysis
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        rolloff_threshold = np.percentile(spectral_rolloff, 30)
        
        # 5. MFCC Analysis (cho voice characteristics)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)[0]
        
        # Detect voice frames
        voice_frames = []
        for i in range(len(rms)):
            # Điều kiện phát hiện voice
            rms_ok = rms[i] > rms_threshold
            centroid_ok = voice_low < spectral_centroids[i] < voice_high
            zcr_ok = zcr_low < zcr[i] < zcr_high
            rolloff_ok = spectral_rolloff[i] > rolloff_threshold
            
            # Cần ít nhất 2/4 điều kiện
            if sum([rms_ok, centroid_ok, zcr_ok, rolloff_ok]) >= 2:
                voice_frames.append(i)
        
        return self._frames_to_segments(voice_frames, sr)
        
    except Exception as e:
        logger.warning(f"Improved voice detection failed: {e}")
        return []
