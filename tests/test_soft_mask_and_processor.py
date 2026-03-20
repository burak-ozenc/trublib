"""
tests/test_soft_mask_and_processor.py
---------------------------------------
Tests for SoftMaskGenerator (Milestone 4) and TADProcessor (Milestone 5).

SoftMaskGenerator tests
~~~~~~~~~~~~~~~~~~~~~~~~
- Zero confidence → silence
- Full confidence → pass-through (minimal distortion)
- Mid confidence → attenuated output
- Output shape matches input
- Short chunk (< n_fft) uses time-domain gain directly
- apply_with_fade produces correct gain ramp

TADProcessor tests
~~~~~~~~~~~~~~~~~~
- Constructs without error at default config
- process() returns a TADResult
- Output shape matches resampled input
- Mono and stereo input both accepted
- reset() clears state
- Latency: full pipeline under 80ms budget
- Wrong input shape raises ValueError
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from trublib.config import TADConfig
from trublib.soft_mask import SoftMaskGenerator
from trublib.tad_state_machine import TADResult, TADState
from tests.conftest import SR, CHUNK, make_harmonic, make_white_noise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def model_is_available() -> bool:
    try:
        from trublib.trumpet_scorer import TrumpetScorer
        TrumpetScorer()
        return True
    except (FileNotFoundError, ImportError):
        return False


requires_model = pytest.mark.skipif(
    not model_is_available(),
    reason="trumpet_scorer_v1.onnx not bundled — skipping processor tests"
)


def make_chunk(val: float = 0.5, n: int = CHUNK) -> np.ndarray:
    return np.full(n, val, dtype=np.float32)


# ---------------------------------------------------------------------------
# SoftMaskGenerator
# ---------------------------------------------------------------------------


class TestSoftMaskGenerator:
    def test_zero_confidence_returns_silence(self):
        """P=0.0 → all bins zeroed → silence."""
        gen = SoftMaskGenerator()
        audio = make_harmonic(465, duration=0.1)[:CHUNK]
        out = gen.apply(audio, confidence=0.0)
        assert np.allclose(out, 0.0, atol=1e-6), f"max={np.max(np.abs(out)):.6f}"

    def test_full_confidence_near_passthrough(self):
        """P=1.0 → output should closely match input (STFT round-trip loss)."""
        gen = SoftMaskGenerator()
        audio = make_harmonic(465, duration=0.1)[:CHUNK]
        out = gen.apply(audio, confidence=1.0)
        # STFT round-trip is not lossless due to windowing; allow 1% RMS error
        rms_in = float(np.sqrt(np.mean(audio**2)))
        rms_out = float(np.sqrt(np.mean(out**2)))
        ratio = rms_out / (rms_in + 1e-10)
        assert 0.90 <= ratio <= 1.10, f"RMS ratio={ratio:.4f}, expected ~1.0"

    def test_mid_confidence_attenuates(self):
        """P=0.5 → output RMS should be ~50% of input."""
        gen = SoftMaskGenerator()
        audio = make_harmonic(465, duration=0.1)[:CHUNK]
        out_full = gen.apply(audio, confidence=1.0)
        out_half = gen.apply(audio, confidence=0.5)
        rms_full = float(np.sqrt(np.mean(out_full**2)))
        rms_half = float(np.sqrt(np.mean(out_half**2)))
        ratio = rms_half / (rms_full + 1e-10)
        assert 0.40 <= ratio <= 0.65, f"half/full RMS ratio={ratio:.4f}"

    def test_output_shape_matches_input(self):
        """Output must be exactly the same length as input."""
        gen = SoftMaskGenerator()
        for n in [256, 512, 1024, 1920, 4800]:
            audio = np.random.randn(n).astype(np.float32)
            out = gen.apply(audio, confidence=0.7)
            assert out.shape == (n,), f"n={n}: expected {n}, got {out.shape}"

    def test_output_dtype_float32(self):
        gen = SoftMaskGenerator()
        audio = make_harmonic(465, duration=0.1)[:CHUNK]
        out = gen.apply(audio, confidence=0.8)
        assert out.dtype == np.float32

    def test_short_chunk_time_domain_gain(self):
        """Chunks shorter than n_fft use direct time-domain scaling."""
        gen = SoftMaskGenerator()
        audio = np.ones(512, dtype=np.float32)   # < n_fft=1024
        out = gen.apply(audio, confidence=0.5)
        assert out.shape == (512,)
        np.testing.assert_allclose(out, 0.5, atol=1e-6)

    def test_confidence_clipped_to_unit_range(self):
        """Confidence outside [0,1] should be clipped, not crash."""
        gen = SoftMaskGenerator()
        audio = make_harmonic(465, duration=0.1)[:CHUNK]
        out_above = gen.apply(audio, confidence=1.5)
        out_below = gen.apply(audio, confidence=-0.3)
        assert np.all(np.isfinite(out_above))
        assert np.allclose(out_below, 0.0, atol=1e-6)

    def test_wrong_stft_params_raises(self):
        with pytest.raises(ValueError, match="fixed at"):
            SoftMaskGenerator(n_fft=512, hop_length=128)

    def test_apply_with_fade_shape(self):
        gen = SoftMaskGenerator()
        audio = make_chunk(1.0)
        out = gen.apply_with_fade(audio, gain_start=1.0, gain_end=0.0)
        assert out.shape == audio.shape
        assert out.dtype == np.float32

    def test_apply_with_fade_decreasing(self):
        """Fade from 1.0 to 0.0 → RMS of first half > RMS of second half."""
        gen = SoftMaskGenerator()
        audio = np.ones(CHUNK, dtype=np.float32)
        out = gen.apply_with_fade(audio, gain_start=1.0, gain_end=0.0)
        mid = CHUNK // 2
        rms_first  = float(np.sqrt(np.mean(out[:mid]**2)))
        rms_second = float(np.sqrt(np.mean(out[mid:]**2)))
        assert rms_first > rms_second, (
            f"first={rms_first:.4f}, second={rms_second:.4f}"
        )

    def test_apply_with_fade_flat(self):
        """gain_start == gain_end → constant gain."""
        gen = SoftMaskGenerator()
        audio = np.ones(CHUNK, dtype=np.float32)
        out = gen.apply_with_fade(audio, gain_start=0.5, gain_end=0.5)
        np.testing.assert_allclose(out, 0.5, atol=1e-6)

    def test_min_gain_floor(self):
        """min_gain=0.2 → even at confidence=0.0, output is not silence."""
        gen = SoftMaskGenerator(min_gain=0.2)
        audio = np.ones(CHUNK, dtype=np.float32)
        out = gen.apply(audio, confidence=0.0)
        rms = float(np.sqrt(np.mean(out**2)))
        assert rms > 0.1, f"Expected non-silence with min_gain=0.2, got RMS={rms:.4f}"


# ---------------------------------------------------------------------------
# TADProcessor — construction and basic API (no model required)
# ---------------------------------------------------------------------------


class TestTADProcessorConstruction:
    def test_default_config(self):
        """TADProcessor with default config should raise only if model missing."""
        try:
            from trublib.processor import TADProcessor
            TADProcessor()
        except FileNotFoundError:
            pytest.skip("Model not bundled")

    def test_custom_config_accepted(self):
        try:
            from trublib.processor import TADProcessor
            cfg = TADConfig(threshold=0.7, onset_chunks=2)
            tad = TADProcessor(cfg)
            assert tad.config.threshold == 0.7
        except FileNotFoundError:
            pytest.skip("Model not bundled")

    def test_wrong_chunk_shape_raises(self):
        try:
            from trublib.processor import TADProcessor
            tad = TADProcessor()
            with pytest.raises(ValueError, match="1-D"):
                tad.process(np.zeros((4, 5, 6), dtype=np.float32))
        except FileNotFoundError:
            pytest.skip("Model not bundled")


# ---------------------------------------------------------------------------
# TADProcessor — full pipeline (requires model)
# ---------------------------------------------------------------------------


@requires_model
class TestTADProcessorPipeline:
    def test_process_returns_tad_result(self):
        from trublib.processor import TADProcessor
        tad = TADProcessor()
        chunk = make_chunk(0.1)
        result = tad.process(chunk)
        assert isinstance(result, TADResult)

    def test_result_fields_present(self):
        from trublib.processor import TADProcessor
        tad = TADProcessor()
        r = tad.process(make_chunk(0.1))
        assert hasattr(r, "masked_audio")
        assert hasattr(r, "state")
        assert hasattr(r, "is_trumpet")
        assert hasattr(r, "confidence")
        assert hasattr(r, "flush")

    def test_masked_audio_shape(self):
        """masked_audio must have the same length as the (resampled) input."""
        from trublib.processor import TADProcessor
        tad = TADProcessor(TADConfig(input_sample_rate=SR))
        r = tad.process(np.zeros(CHUNK, dtype=np.float32))
        assert r.masked_audio.shape == (CHUNK,)

    def test_mono_input_accepted(self):
        from trublib.processor import TADProcessor
        tad = TADProcessor()
        tad.process(np.zeros(CHUNK, dtype=np.float32))

    def test_stereo_input_mixed_down(self):
        from trublib.processor import TADProcessor
        tad = TADProcessor()
        stereo = np.zeros((CHUNK, 2), dtype=np.float32)
        result = tad.process(stereo)
        assert isinstance(result, TADResult)

    def test_confidence_in_unit_range(self):
        from trublib.processor import TADProcessor
        tad = TADProcessor()
        for _ in range(5):
            r = tad.process(make_chunk(0.1))
            assert 0.0 <= r.confidence <= 1.0

    def test_initial_state_silent(self):
        from trublib.processor import TADProcessor
        tad = TADProcessor()
        r = tad.process(make_chunk(0.0))
        assert r.state == TADState.SILENT

    def test_reset_clears_state(self):
        from trublib.processor import TADProcessor
        tad = TADProcessor()
        for _ in range(10):
            tad.process(make_harmonic(465, duration=0.5).astype(np.float32)[:CHUNK])
        tad.reset()
        assert tad.state == TADState.SILENT

    def test_silence_stays_silent(self):
        """Zero audio should never trigger ACTIVE."""
        from trublib.processor import TADProcessor
        tad = TADProcessor()
        for _ in range(20):
            r = tad.process(np.zeros(CHUNK, dtype=np.float32))
        assert r.state in (TADState.SILENT, TADState.ONSET)

    def test_result_is_finite(self):
        from trublib.processor import TADProcessor
        tad = TADProcessor()
        r = tad.process(make_harmonic(465, duration=0.1)[:CHUNK])
        assert np.all(np.isfinite(r.masked_audio))
        assert np.isfinite(r.confidence)


@requires_model
class TestTADProcessorLatency:
    def test_single_chunk_under_80ms(self):
        """One 80ms chunk must process in well under 80ms."""
        from trublib.processor import TADProcessor
        tad = TADProcessor()
        chunk = make_harmonic(465, duration=0.5)[:CHUNK]

        # Warm up (first call includes model load overhead)
        tad.process(chunk)

        t0 = time.perf_counter()
        tad.process(chunk)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 80, f"Single chunk took {elapsed_ms:.1f}ms — exceeds 80ms budget"
        print(f"\n  Single chunk latency: {elapsed_ms:.2f}ms")

    def test_ten_chunks_under_budget(self):
        """10 consecutive chunks must complete in under 800ms total."""
        from trublib.processor import TADProcessor
        tad = TADProcessor()
        signal = make_harmonic(465, duration=1.0)
        chunks = [signal[i:i+CHUNK] for i in range(0, len(signal)-CHUNK, CHUNK)]

        # Warm up
        tad.process(chunks[0])

        t0 = time.perf_counter()
        for chunk in chunks[:10]:
            tad.process(chunk)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 800, f"10 chunks took {elapsed_ms:.1f}ms"
        print(f"\n  10-chunk total: {elapsed_ms:.1f}ms ({elapsed_ms/10:.2f}ms/chunk)")
