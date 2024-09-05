import queue
import logging
import threading
import numpy as np
from scipy import signal
import pywt
from sklearn.ensemble import IsolationForest
from typing import List, Set

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class Pulse:
    def __init__(
        self,
        window_size: int = 1024,
        isolation_forest_contamination: float = 0.01,
        spectral_threshold: float = 3.0,
        wavelet_threshold: float = 3.0,
    ) -> None:
        self.window_size: int = window_size
        self.isolation_forest_contamination: float = isolation_forest_contamination
        self.spectral_threshold: float = spectral_threshold
        self.wavelet_threshold: float = wavelet_threshold

        self.data_queue: queue.Queue[float] = queue.Queue()
        self.buffer: List[float] = []
        self.isolation_forest_model: IsolationForest = IsolationForest(
            contamination=isolation_forest_contamination, random_state=42
        )

        self.processing_thread: threading.Thread = threading.Thread(target=self._process_data)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def ingest_data(self, data: float) -> None:
        self.data_queue.put(data)

    def _process_data(self) -> None:
        while True:
            try:
                data: float = self.data_queue.get(timeout=1, block=True)
                self.buffer.append(data)
                if len(self.buffer) >= self.window_size:
                        try:
                            self._detect_anomalies(self.buffer[-self.window_size :])
                            self.buffer = self.buffer[-self.window_size :]  # Keep only the latest window
                        except AnomalyDetectedExceptionError as e:
                            logging.error(f"Anomaly detected: {e}")
            except queue.Empty:
                continue

    def _detect_anomalies(self, data: List[float]) -> None:
        data_array: np.ndarray = np.array(data)

        # Time-domain analysis - Isolation Forest
        isolation_forest_pred: np.ndarray = self.isolation_forest_model.fit_predict(data_array.reshape(-1, 1))
        isolation_forest_anomalies: np.ndarray = np.where(isolation_forest_pred == -1)[0]

        # Frequency-domain analysis - Spectral analysis
        spectral_anomalies: Set[int] = self._spectral_analysis(data_array)

        # Time-frequency analysis - Wavelet transform
        wavelet_anomalies: Set[int] = self._wavelet_analysis(data_array)

        all_anomalies: Set[int] = set(isolation_forest_anomalies) | spectral_anomalies | wavelet_anomalies

        if all_anomalies:
            anomaly_indices: List[int] = sorted(list(all_anomalies))
            raise AnomalyDetectedExceptionError(f"Anomalies detected at indices: {anomaly_indices} - {all_anomalies}")

        # Update Isolation Forest model
        self.isolation_forest_model.fit(data_array.reshape(-1, 1))

    def _spectral_analysis(self, data: np.ndarray) -> Set[int]:
        f: np.ndarray
        pxx: np.ndarray
        f, pxx = signal.welch(data, fs=1.0, nperseg=min(256, len(data)))
        mean_psd: float = np.mean(pxx)
        std_psd: float = np.std(pxx)
        threshold: float = mean_psd + self.spectral_threshold * std_psd

        anomalous_freqs: np.ndarray = f[pxx > threshold]
        anomalies: List[int] = []
        for freq in anomalous_freqs:
            sin_wave: np.ndarray = np.sin(2 * np.pi * freq * np.arange(len(data)))
            correlation: np.ndarray = np.correlate(data, sin_wave, mode="same")
            anomalies.extend(
                np.where(np.abs(correlation) > np.mean(np.abs(correlation)) + 2 * np.std(np.abs(correlation)))[0]
            )

        return set(anomalies)

    def _wavelet_analysis(self, data: np.ndarray) -> Set[int]:
        coeffs: List[np.ndarray] = pywt.wavedec(data, "db4", level=5)
        anomalies: List[int] = []
        for level, coeff in enumerate(coeffs):
            mean_coeff: float = np.mean(coeff)
            std_coeff: float = np.std(coeff)
            threshold: float = mean_coeff + self.wavelet_threshold * std_coeff
            level_anomalies: np.ndarray = np.where(np.abs(coeff) > threshold)[0]
            scale_factor: int = 2**level
            anomalies.extend(scale_factor * level_anomalies)

        return set(anomalies)
    
    def __del__(self):
        self.stop_thread = True
        self.processing_thread.join()


class AnomalyDetectedExceptionError(Exception):
    """Custom exception raised when anomalies are detected."""
    pass
