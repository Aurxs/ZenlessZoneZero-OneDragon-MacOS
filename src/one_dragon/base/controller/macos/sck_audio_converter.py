"""复制 ScreenCaptureKit 音频，并保留跨回调的重采样滤波状态。"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, isfinite
from typing import Any

import numpy as np
from scipy.signal import firwin, lfilter


@dataclass(frozen=True)
class CapturedAudio:
    """不再引用原生缓冲的单声道音频，时间来自样本的呈现时间。"""

    samples: np.ndarray
    sample_rate: int
    presentation_time: float


def copy_audio_sample(sample: Any) -> CapturedAudio:
    """读取真实 ASBD 和 AudioBufferList，不假设平面或交错排列。"""
    import CoreAudio
    import CoreMedia

    description = CoreMedia.CMSampleBufferGetFormatDescription(sample)
    if description is None:
        raise ValueError('音频样本缺少格式')
    fmt = CoreMedia.CMAudioFormatDescriptionGetStreamBasicDescription(description)
    if fmt is None:
        raise ValueError('音频样本缺少 PCM 描述')
    channels = int(fmt.mChannelsPerFrame)
    rate = float(fmt.mSampleRate)
    flags = int(fmt.mFormatFlags)
    if (int(fmt.mFormatID) != CoreAudio.kAudioFormatLinearPCM
            or not flags & CoreAudio.kAudioFormatFlagIsFloat
            or flags & CoreAudio.kAudioFormatFlagIsBigEndian
            or int(fmt.mBitsPerChannel) != 32
            or channels not in (1, 2) or rate not in (32000, 48000)):
        raise ValueError('窗口音频必须是 32/48 kHz、单/双声道、小端浮点 PCM')
    count = int(CoreMedia.CMSampleBufferGetNumSamples(sample))
    timestamp = float(CoreMedia.CMTimeGetSeconds(CoreMedia.CMSampleBufferGetPresentationTimeStamp(sample)))
    if count <= 0 or count > int(rate) or not isfinite(timestamp):
        raise ValueError('音频样本长度或时间无效')
    # 首次只查询大小，随后由 CoreAudio 分配可容纳全部声道的列表。
    _, size, _ = CoreMedia.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sample, None, None, 0, None, None, 0, None,
    )
    buffers = CoreAudio.AudioBufferList(channels)
    if not 0 < size <= 8 + 16 * channels:
        raise ValueError('音频缓冲列表大小无效')
    status, _, retained_block = CoreMedia.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sample, None, buffers, size, None, None, 0, None,
    )
    if status != 0 or retained_block is None:
        raise ValueError(f'无法读取音频缓冲: {status}')
    planes: list[np.ndarray] = []
    for buffer in buffers:
        if buffer.mNumberChannels == 0:
            continue
        buffer_channels = int(buffer.mNumberChannels)
        expected = count * buffer_channels * 4
        if buffer.mData is None or buffer.mDataByteSize != expected:
            raise ValueError('音频缓冲长度与格式不符')
        # retained_block 必须活到复制结束，不能把原生内存带出回调。
        values = np.frombuffer(buffer.mData, dtype='<f4', count=count * buffer_channels)
        planes.append(values.reshape(count, buffer_channels).astype(np.float64))
    if not planes or sum(plane.shape[1] for plane in planes) != channels:
        raise ValueError('音频缓冲声道不完整')
    mono = np.concatenate(planes, axis=1).mean(axis=1)
    if not np.isfinite(mono).all():
        raise ValueError('音频样本包含非有限数值')
    return CapturedAudio(mono, int(rate), timestamp)


class StreamingAudioConverter:
    """连续多相滤波；块边界不补零、不重置相位，暂停后显式重置。"""

    def __init__(self, source_rate: int, target_rate: int = 32000) -> None:
        """只接受窗口采集实际支持的采样率。"""
        if source_rate not in (32000, 48000) or target_rate != 32000:
            raise ValueError('窗口音频转换仅支持 32/48 kHz 到 32 kHz')
        common = gcd(source_rate, target_rate)
        self._up: int = target_rate // common
        self._down: int = source_rate // common
        factor = max(self._up, self._down)
        self._taps: np.ndarray = (
            np.ones(1) if factor == 1
            else firwin(20 * factor + 1, 1 / factor, window=('kaiser', 5.0)) * self._up
        )
        self._state: np.ndarray = np.zeros(self._taps.size - 1)
        self._phase: int = 0

    def push(self, mono: np.ndarray) -> np.ndarray:
        """返回本块产生的 32 kHz 样本，保留滤波尾部到下一块。"""
        if mono.ndim != 1 or not np.isfinite(mono).all():
            raise ValueError('重采样输入必须是有限的单声道样本')
        if mono.size == 0:
            return np.empty(0, dtype=np.float64)
        if self._up == self._down:
            return mono.astype(np.float64, copy=True)
        expanded = np.zeros(mono.size * self._up)
        expanded[::self._up] = mono
        filtered, self._state = lfilter(self._taps, [1.0], expanded, zi=self._state)
        output = filtered[self._phase::self._down].copy()
        self._phase = (self._phase - expanded.size) % self._down
        return output
