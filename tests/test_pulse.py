import pytest
import numpy as np
import time
from src.pulse import Pulse, AnomalyDetectedExceptionError


def simulate_signal_stream():
    """Simulates a continuous signal stream with periodic anomalies."""
    t = 0
    while True:
        # Normal signal with some harmonics
        value = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.1)

        # Introduce anomalies
        if t % 100 == 0:
            value += 2  # Spike anomaly
        elif 300 < t < 320:
            value += np.random.normal(0, 0.5)  # Burst noise
        elif t % 500 == 0:
            value *= 0.1  # Sudden amplitude change
        yield value
        t += 1


@pytest.fixture
def pulse():
    pulse_instance = Pulse(window_size=1024, isolation_forest_contamination=0.01, spectral_threshold=3, wavelet_threshold=3)

    yield pulse_instance
    pulse_instance.stop_thread = True

def test_signal_anomalies(pulse: Pulse, caplog: pytest.LogCaptureFixture):
    stream = simulate_signal_stream()

    # Run for a larger number of iterations to simulate data ingestion
    for i, data in enumerate(stream):
        pulse.ingest_data(data)
        if i % 100 == 0:
            print(f"Processed {i} data points")

        # After a sufficient number of data points, force anomaly checking
        if i >= 2000:
            break

    # Allow some time for the processing thread to process data
    time.sleep(1)

    assert "Anomaly detected" in caplog.text
