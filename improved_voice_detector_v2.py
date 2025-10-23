
def _detect_voice_activity_improved_v2(self, audio: np.ndarray, sr: int) -> List[Dict]:
    """Phát hiện voice activity cải thiện v2 - dành cho file karaoke"""
    try:
        # 1. RMS Energy Analysis với threshold thấp hơn
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        # Sử dụng percentile thấp hơn để phát hiện voice nhẹ
        rms_threshold = np.percentile(rms, 15)  # Thấp hơn nữa
        
        # 2. Spectral Centroid Analysis
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Voice frequency range (100-3500 Hz) - mở rộng hơn
        voice_low = 100
        voice_high = 3500
        
        # 3. Zero Crossing Rate Analysis
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Voice có ZCR trung bình
        zcr_low = np.percentile(zcr, 5)   # Thấp hơn
        zcr_high = np.percentile(zcr, 95) # Cao hơn
        
        # 4. Spectral Rolloff Analysis
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        rolloff_threshold = np.percentile(spectral_rolloff, 20)  # Thấp hơn
        
        # 5. MFCC Analysis (cho voice characteristics)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)[0]
        
        # 6. Spectral Bandwidth Analysis
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        bandwidth_threshold = np.percentile(spectral_bandwidth, 25)
        
        # Detect voice frames
        voice_frames = []
        for i in range(len(rms)):
            # Điều kiện phát hiện voice - cần ít nhất 2/6 điều kiện
            rms_ok = rms[i] > rms_threshold
            centroid_ok = voice_low < spectral_centroids[i] < voice_high
            zcr_ok = zcr_low < zcr[i] < zcr_high
            rolloff_ok = spectral_rolloff[i] > rolloff_threshold
            bandwidth_ok = spectral_bandwidth[i] > bandwidth_threshold
            
            # Cần ít nhất 2/5 điều kiện
            if sum([rms_ok, centroid_ok, zcr_ok, rolloff_ok, bandwidth_ok]) >= 2:
                voice_frames.append(i)
        
        return self._frames_to_segments(voice_frames, sr)
        
    except Exception as e:
        logger.warning(f"Improved voice detection v2 failed: {e}")
        return []
