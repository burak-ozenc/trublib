from trublib.feature_extractor import FeatureExtractor
from trublib.frame_manager import FrameManager
from trublib.normalizer import TwoStageNormalizer
from scipy.io import wavfile
import numpy as np

SR, CHUNK = 24_000, 1_920
sr, data = wavfile.read("data/diagnose/trumpet_trumpet_scale_jesus_recordings_3_0.wav")
samples = data.astype(np.float32) / 32768.0

norm = TwoStageNormalizer()
fm   = FrameManager()
ex   = FeatureExtractor(sr=SR)

print(f"{'chunk':>5}  {'time':>6}  {'rms_db':>7}  {'flatness':>9}  {'f0_hz':>7}  {'hnr_db':>7}  {'pitch_sal':>10}")
for i in range(len(samples) // CHUNK):
    chunk = samples[i*CHUNK:(i+1)*CHUNK]
    normalised, rms_db = norm.process(chunk)
    frames = fm.push(normalised)
    fvs = [ex.extract(f, rms_db_override=rms_db) for f in frames]
    if not fvs: continue
    v = np.stack([fv.to_vector() for fv in fvs]).mean(axis=0)
    print(f"{i:>5}  {i*0.08:>5.2f}s  {rms_db:>7.1f}  {v[1]:>9.3f}  {v[4]*2000:>7.0f}  {v[6]*50-10:>7.2f}  {v[5]:>10.3f}")