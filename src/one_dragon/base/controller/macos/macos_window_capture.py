"""只读采集指定 macOS 窗口，不激活窗口或产生桌面输入。"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np

from one_dragon.base.controller.macos.sck_audio_converter import (
    CapturedAudio,
    copy_audio_sample,
)
from one_dragon.base.controller.macos.sck_frame_converter import (
    CapturedFrame,
    bgra_buffer_to_rgb,
    capture_timestamps,
    normalize_game_frame,
)


class MacCaptureError(RuntimeError):
    """窗口选择、权限或原生采集失败。"""


@dataclass(frozen=True)
class MacWindow:
    """一次枚举得到的窗口身份；尺寸是逻辑点，窗口 ID 不持久化。"""

    bundle_id: str
    pid: int
    window_id: int
    title: str
    width: int
    height: int
    on_screen: bool = True


@dataclass(frozen=True)
class CaptureDiagnostics:
    """只读采集状态；回调到达时间与原生有效观察时间分开记录。"""

    running: bool
    active: bool
    requires_pixel_change: bool
    content_verified: bool
    sequence: int
    sample_status: int | None
    sample_counts: dict[int, int]
    callback_age_seconds: float | None
    content_age_seconds: float | None
    observation_age_seconds: float | None
    error: str | None


def _initialize_appkit(appkit: Any) -> None:
    """让纯 Python 进程连接 WindowServer，保留已有 Qt 应用的策略。"""
    if appkit.NSApp() is not None:
        return
    if threading.current_thread() is not threading.main_thread():
        raise MacCaptureError('纯 Python 进程必须在主线程首次初始化窗口采集')
    application = appkit.NSApplication.sharedApplication()
    # 命令行进程可能拒绝修改策略；连接 WindowServer 只依赖 sharedApplication。
    # 不调用会激活应用的 finishLaunching 或 run。
    application.setActivationPolicy_(appkit.NSApplicationActivationPolicyProhibited)


@lru_cache
def _native_api() -> Any:
    """实际使用时才加载 macOS 框架，保持模块在 Windows 上可导入。"""
    if sys.platform != 'darwin':
        raise MacCaptureError('窗口采集仅支持 macOS')
    import AppKit
    import CoreMedia
    import dispatch
    import Foundation
    import objc
    import Quartz
    import ScreenCaptureKit

    _initialize_appkit(AppKit)
    if Quartz.CGMainDisplayID() == 0:
        raise MacCaptureError('当前进程没有可用的图形登录会话，未启动窗口采集')

    class SckWindowOutput(Foundation.NSObject, protocols=[
        objc.protocolNamed('SCStreamOutput'), objc.protocolNamed('SCStreamDelegate'),
    ]):
        """把原生串行队列回调交给当前采集实例。"""

        _capture: MacWindowCapture | None

        @objc.python_method
        def bind(self, capture: MacWindowCapture | None) -> None:
            """绑定或解除拥有者，停止时打破原生对象引用。"""
            self._capture = capture

        def stream_didOutputSampleBuffer_ofType_(self, stream: Any, sample: Any, output_type: int) -> None:
            """接收原生视频回调；此处不执行识别或输入。"""
            if self._capture is not None:
                self._capture._receive_sample(stream, sample, output_type)

        def stream_didStopWithError_(self, stream: Any, error: Any) -> None:
            """采集异常后立即清空旧帧。"""
            if self._capture is not None:
                self._capture._fail_stream(stream, f'窗口采集已停止: {error}')

        def streamDidBecomeInactive_(self, stream: Any) -> None:
            """共享内容暂不可用时清帧，保留流以接收系统恢复通知。"""
            if self._capture is not None:
                self._capture._set_stream_active(stream, False)

        def streamDidBecomeActive_(self, stream: Any) -> None:
            """恢复通知本身不是有效帧，仍等待新的像素缓冲。"""
            if self._capture is not None:
                self._capture._set_stream_active(stream, True)

    return (CoreMedia, dispatch, Foundation, Quartz, ScreenCaptureKit, SckWindowOutput)


class MacWindowCapture:
    """独立 SCStream 与单个最新帧槽，可由 Qt 或普通 Python 进程使用。"""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        """创建采集状态；构造时不申请权限或启动采集。"""
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('原生采集等待时间必须大于 0')
        self._timeout_seconds: float = timeout_seconds
        self._lock: threading.Lock = threading.Lock()
        self._stream: Any = None
        self._output: Any = None
        self._queue: Any = None
        self._configuration: Any = None
        self._audio_control_lock: threading.RLock = threading.RLock()
        self._audio_queue: Any = None
        self._audio_callback: Callable[[CapturedAudio], None] | None = None
        self._audio_error_callback: Callable[[RuntimeError], None] | None = None
        self._audio_start_pts: float | None = None
        self._last_audio_error: str | None = None
        self._running: bool = False
        self._active: bool = True
        self._activity_generation: int = 0
        self._requires_pixel_change: bool = False
        self._content_verified: bool = False
        self._last_sample_monotonic: float | None = None
        self._sample_status: int | None = None
        self._sample_counts: dict[int, int] = {}
        self._latest: CapturedFrame | None = None
        self._last_observation_monotonic: float | None = None
        self._freshness_error: str | None = None
        self._raw_size: tuple[int, int] | None = None
        self._sequence: int = 0
        self._last_error: str | None = None
        self._content_rect: tuple[int, int, int, int] | None = None
        self._top_inset_pixels: int = 0
        self._top_inset_points: float = 0
        self._auto_top_inset: bool = False
        self.window: MacWindow | None = None

    @property
    def last_error(self) -> str | None:
        """返回最近一次失败原因，供控制器在无帧时说明原因。"""
        with self._lock:
            return self._last_error or self._freshness_error

    @property
    def audio_error(self) -> str | None:
        """声音错误独立记录，不使有效视频帧和窗口就绪状态失效。"""
        with self._lock:
            return self._last_audio_error

    @property
    def raw_size(self) -> tuple[int, int] | None:
        """返回原生像素缓冲尺寸，不把逻辑点误报为像素。"""
        with self._lock:
            return self._raw_size

    def diagnostics(self) -> CaptureDiagnostics:
        """读取流状态，不枚举窗口、不更新任何有效帧时间。"""
        with self._lock:
            now = time.monotonic()
            return CaptureDiagnostics(
                running=self._running,
                active=self._active,
                requires_pixel_change=self._requires_pixel_change,
                content_verified=self._content_verified,
                sequence=self._sequence,
                sample_status=self._sample_status,
                sample_counts=dict(self._sample_counts),
                callback_age_seconds=(
                    None if self._last_sample_monotonic is None else now - self._last_sample_monotonic
                ),
                content_age_seconds=None if self._latest is None else now - self._latest.monotonic_time,
                observation_age_seconds=(
                    None if self._last_observation_monotonic is None else now - self._last_observation_monotonic
                ),
                error=self._last_error or self._freshness_error,
            )

    def _wait(self, invoke: Callable[[Callable[..., None]], None], operation: str) -> tuple[Any, ...]:
        """有界等待原生异步完成；主线程等待时仍处理 Cocoa 回调。"""
        foundation = _native_api()[2]
        completed = threading.Event()
        result: tuple[Any, ...] = ()

        def completion(*values: Any) -> None:
            """保存完成结果并唤醒等待者。"""
            nonlocal result
            result = values
            completed.set()

        invoke(completion)
        deadline = time.monotonic() + self._timeout_seconds
        while not completed.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MacCaptureError(f'{operation}超时，请检查屏幕录制权限与目标窗口')
            interval = min(0.01, remaining)
            if threading.current_thread() is threading.main_thread():
                foundation.NSRunLoop.currentRunLoop().runUntilDate_(
                    foundation.NSDate.dateWithTimeIntervalSinceNow_(interval),
                )
            else:
                completed.wait(interval)
        return result

    def _shareable_windows(self) -> list[Any]:
        """枚举所有 Space 的窗口，不以当前桌面可见性过滤目标。"""
        sck = _native_api()[4]
        content, error = self._wait(
            lambda completion: sck.SCShareableContent
            .getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                True, False, completion,
            ),
            '枚举窗口',
        )
        if error is not None or content is None:
            raise MacCaptureError(f'无法枚举窗口，请授予当前 Python 进程屏幕录制权限: {error}')
        return list(content.windows())

    def list_windows(self, bundle_id: str) -> list[MacWindow]:
        """只枚举指定 bundle ID 的窗口，不使用当前活动应用作为目标。"""
        if not bundle_id:
            raise ValueError('必须明确指定目标应用的 bundle ID')
        windows: list[MacWindow] = []
        for window in self._shareable_windows():
            app = window.owningApplication()
            if app is None or app.bundleIdentifier() != bundle_id:
                continue
            rect = window.frame()
            windows.append(MacWindow(
                bundle_id=bundle_id,
                pid=int(app.processID()),
                window_id=int(window.windowID()),
                title=str(window.title() or ''),
                width=round(rect.size.width),
                height=round(rect.size.height),
                on_screen=bool(window.isOnScreen()),
            ))
        return windows

    def start(
        self,
        window: MacWindow,
        *,
        fps: int = 30,
        content_rect: tuple[int, int, int, int] | None = None,
        top_inset_pixels: int = 0,
        top_inset_points: float = 0,
        auto_top_inset: bool = False,
    ) -> None:
        """绑定窗口；自动、像素、逻辑点裁剪和固定游戏区只能选一种。"""
        if not 1 <= fps <= 120:
            raise ValueError('采集帧率必须在 1 到 120 之间')
        if type(top_inset_pixels) is not int or top_inset_pixels < 0:
            raise ValueError('顶部裁剪像素必须是非负整数')
        if not math.isfinite(top_inset_points) or top_inset_points < 0:
            raise ValueError('顶部裁剪逻辑点必须是非负有限数')
        if top_inset_points and top_inset_pixels:
            raise ValueError('不能同时指定顶部裁剪像素和逻辑点')
        if (top_inset_pixels or top_inset_points) and content_rect is not None:
            raise ValueError('顶部裁剪和固定游戏内容区不能同时使用')
        if type(auto_top_inset) is not bool:
            raise ValueError('自动顶部裁剪选项必须是布尔值')
        if auto_top_inset and (top_inset_pixels or top_inset_points or content_rect is not None):
            raise ValueError('自动顶部裁剪不能同时使用手动裁剪')
        self.stop()
        cm, dispatch, _, quartz, sck, output_class = _native_api()
        selected = None
        for candidate in self._shareable_windows():
            app = candidate.owningApplication()
            if (app is not None and int(candidate.windowID()) == window.window_id
                    and int(app.processID()) == window.pid
                    and app.bundleIdentifier() == window.bundle_id):
                selected = candidate
                break
        if selected is None:
            raise MacCaptureError('目标窗口已消失或重建，需要重新枚举并选择窗口')

        content_filter = sck.SCContentFilter.alloc().initWithDesktopIndependentWindow_(selected)
        rect = content_filter.contentRect()
        scale = content_filter.pointPixelScale()
        config = sck.SCStreamConfiguration.alloc().init()
        config.setWidth_(round(rect.size.width * scale))
        config.setHeight_(round(rect.size.height * scale))
        config.setPixelFormat_(quartz.kCVPixelFormatType_32BGRA)
        config.setMinimumFrameInterval_(cm.CMTimeMake(1, fps))
        config.setQueueDepth_(3)
        config.setShowsCursor_(False)
        config.setCapturesAudio_(False)
        config.setSampleRate_(48000)
        config.setChannelCount_(2)
        config.setExcludesCurrentProcessAudio_(True)
        config.setCaptureMicrophone_(False)
        config.setIncludeChildWindows_(False)
        config.setIgnoreShadowsSingleWindow_(True)
        config.setIgnoreGlobalClipSingleWindow_(True)
        config.setPreservesAspectRatio_(True)
        config.setScalesToFit_(False)
        output = output_class.alloc().init()
        output.bind(self)
        queue = dispatch.dispatch_queue_create(b'one_dragon.window_capture', None)
        stream = sck.SCStream.alloc().initWithFilter_configuration_delegate_(content_filter, config, output)
        success, error = stream.addStreamOutput_type_sampleHandlerQueue_error_(
            output, sck.SCStreamOutputTypeScreen, queue, None,
        )
        if not success:
            output.bind(None)
            raise MacCaptureError(f'无法注册窗口采集输出: {error}')
        with self._lock:
            self._stream, self._output, self._queue = stream, output, queue
            self._configuration = config
            self._running = True
            self._active = bool(selected.isOnScreen())
            self._requires_pixel_change = not self._active
            self._content_verified = False
            self._last_error = None
            self._content_rect = content_rect
            self._top_inset_pixels = top_inset_pixels
            self._top_inset_points = top_inset_points
            self._auto_top_inset = auto_top_inset
            self.window = window
        try:
            error, = self._wait(stream.startCaptureWithCompletionHandler_, '启动窗口采集')
            if error is not None:
                raise MacCaptureError(f'无法启动窗口采集，请检查屏幕录制权限: {error}')
        except Exception:
            self.stop()
            raise

    def start_audio(
        self,
        callback: Callable[[CapturedAudio], None],
        error_callback: Callable[[RuntimeError], None],
    ) -> None:
        """在同一目标流上订阅游戏应用音频，不另录桌面或麦克风。"""
        with self._audio_control_lock:
            with self._lock:
                stream, output, config = self._stream, self._output, self._configuration
                if not self._running or stream is None or config is None:
                    raise MacCaptureError('游戏窗口尚未开始采集，声音闪避保持画面识别')
                if self._audio_queue is not None:
                    raise MacCaptureError('当前窗口已有声音订阅，请先停止上一场声音识别')
            cm, dispatch, _, _, sck, _ = _native_api()
            started_pts = float(cm.CMTimeGetSeconds(cm.CMClockGetTime(cm.CMClockGetHostTimeClock())))
            queue = dispatch.dispatch_queue_create(b'one_dragon.window_audio', None)
            success, error = stream.addStreamOutput_type_sampleHandlerQueue_error_(
                output, sck.SCStreamOutputTypeAudio, queue, None,
            )
            if not success:
                raise MacCaptureError(f'无法注册游戏声音输出: {error}')
            with self._lock:
                self._audio_queue = queue
                self._audio_callback = callback
                self._audio_error_callback = error_callback
                self._audio_start_pts = started_pts
                self._last_audio_error = None
            try:
                audio_config = config.copy()
                audio_config.setCapturesAudio_(True)
                error, = self._wait(
                    lambda completion: stream.updateConfiguration_completionHandler_(audio_config, completion),
                    '启动游戏声音采集',
                )
                if error is not None:
                    raise MacCaptureError(f'无法启动游戏声音采集: {error}')
                with self._lock:
                    self._configuration = audio_config
            except Exception:
                self.stop_audio()
                raise

    def stop_audio(self) -> None:
        """先断开音频消费者，再关闭声音输出；保留现有画面采集。"""
        with self._audio_control_lock:
            with self._lock:
                stream, output, config = self._stream, self._output, self._configuration
                registered = self._audio_queue is not None
                self._audio_callback = self._audio_error_callback = None
                self._audio_start_pts = None
                self._audio_queue = None
                running = self._running
            if not registered or stream is None:
                return
            sck = _native_api()[4]
            if running and config is not None:
                try:
                    quiet_config = config.copy()
                    quiet_config.setCapturesAudio_(False)
                    error, = self._wait(
                        lambda completion: stream.updateConfiguration_completionHandler_(quiet_config, completion),
                        '停止游戏声音采集',
                    )
                    if error is None:
                        with self._lock:
                            self._configuration = quiet_config
                    else:
                        with self._lock:
                            self._last_audio_error = f'关闭游戏声音配置失败: {error}'
                except Exception as error:
                    # 移除声音输出仍须继续，不能因为配置更新失败保留消费者。
                    with self._lock:
                        self._last_audio_error = f'关闭游戏声音配置失败: {error}'
            try:
                success, error = stream.removeStreamOutput_type_error_(output, sck.SCStreamOutputTypeAudio, None)
                if not success:
                    with self._lock:
                        self._last_audio_error = f'移除游戏声音输出失败: {error}'
            except Exception as error:
                with self._lock:
                    self._last_audio_error = f'移除游戏声音输出失败: {error}'

    def _fail_audio(self, stream: Any, message: str) -> None:
        """音频失败只通知一次，不使有效画面失效。"""
        with self._lock:
            if stream != self._stream:
                return
            callback = self._audio_error_callback
            self._audio_callback = self._audio_error_callback = None
            if callback is not None:
                self._last_audio_error = message
        if callback is not None:
            try:
                callback(MacCaptureError(message))
            except Exception as error:
                # 界面通知失败也不能中断原生流的停止和资源释放。
                with self._lock:
                    self._last_audio_error = f'{message}；通知声音消费者失败: {error}'

    def _set_stream_active(self, stream: Any, active: bool) -> None:
        """暂时失活不销毁流；恢复只接受新的 Complete，不复用失活前像素。"""
        with self._lock:
            if not self._running or stream != self._stream or self._active == active:
                return
            self._active = active
            self._activity_generation += 1
            self._requires_pixel_change = not active
            self._content_verified = False
            self._latest = None
            self._last_observation_monotonic = None
            self._freshness_error = None
            self._last_error = (
                '窗口采集已恢复活动，等待新的完整画面' if active
                else '共享窗口暂不可采集，等待系统恢复；未切换桌面'
            )
        if not active:
            self._fail_audio(stream, '游戏窗口暂不可采集，已停止声音闪避并清空录音')

    def _fail_stream(self, stream: Any, error: str) -> None:
        """异常、窗口关闭和权限撤销后立即使旧帧失效。"""
        with self._lock:
            if stream != self._stream:
                return
            self._running = False
            self._active = False
            self._content_verified = False
            self._latest = None
            self._last_observation_monotonic = None
            self._freshness_error = None
            self._last_error = error
        self._fail_audio(stream, error)

    def _receive_sample(self, stream: Any, sample: Any, output_type: int) -> None:
        """复制一帧原生像素后覆盖最新帧，不保留原生缓冲的引用。"""
        cm, _, _, quartz, sck, _ = _native_api()
        with self._lock:
            if not self._running or stream != self._stream:
                return
            activity_generation = self._activity_generation
        if output_type == getattr(sck, 'SCStreamOutputTypeAudio', 1):
            with self._lock:
                callback = self._audio_callback
            if callback is None:
                return
            try:
                if not cm.CMSampleBufferIsValid(sample) or not cm.CMSampleBufferDataIsReady(sample):
                    raise MacCaptureError('游戏声音样本尚未就绪')
                audio = copy_audio_sample(sample)
                with self._lock:
                    if (not self._running or stream != self._stream
                            or activity_generation != self._activity_generation
                            or callback is not self._audio_callback):
                        return
                    started_pts = self._audio_start_pts
                # 停止再订阅仍使用同一个 SCStream；按原生样本时间拒绝上一轮排队的数据。
                if started_pts is None or audio.presentation_time < started_pts:
                    return
                now_pts = float(cm.CMTimeGetSeconds(cm.CMClockGetTime(cm.CMClockGetHostTimeClock())))
                age = now_pts - audio.presentation_time
                if not -0.05 <= age <= 0.5:
                    raise MacCaptureError('游戏声音样本时间已过期或不属于当前主机时钟')
                callback(audio)
            except Exception as error:
                self._fail_audio(stream, f'读取游戏声音失败: {error}')
            return
        if output_type != sck.SCStreamOutputTypeScreen or not cm.CMSampleBufferIsValid(sample):
            return
        try:
            attachments = cm.CMSampleBufferGetSampleAttachmentsArray(sample, False)
            if not attachments:
                raise MacCaptureError('采集帧缺少元数据')
            metadata = attachments[0]
            status = metadata.get(sck.SCStreamFrameInfoStatus)
            with self._lock:
                if not self._running or stream != self._stream:
                    return
                self._last_sample_monotonic = time.monotonic()
                self._sample_status = None if status is None else int(status)
                if self._sample_status is not None:
                    self._sample_counts[self._sample_status] = self._sample_counts.get(self._sample_status, 0) + 1
            if status == sck.SCFrameStatusIdle:
                # Idle 表示画面没有变化，不生成新像素；只能更新该流的观察时间。
                try:
                    _, observed_at = capture_timestamps(
                        int(metadata[sck.SCStreamFrameInfoDisplayTime]),
                        int(quartz.CVGetCurrentHostTime()),
                        float(quartz.CVGetHostClockFrequency()),
                        epoch_now=time.time(), monotonic_now=time.monotonic(),
                    )
                except (KeyError, TypeError, ValueError):
                    return
                with self._lock:
                    if (not self._running or stream != self._stream or self._latest is None
                            or activity_generation != self._activity_generation
                            or self._requires_pixel_change):
                        return
                    previous = self._last_observation_monotonic
                    if observed_at >= max(self._latest.monotonic_time, previous or self._latest.monotonic_time):
                        self._last_observation_monotonic = observed_at
                        self._freshness_error = None
                return
            if status != sck.SCFrameStatusComplete:
                if status == sck.SCFrameStatusStopped:
                    self._fail_stream(stream, '窗口采集已停止')
                    return
                with self._lock:
                    if self._running and stream == self._stream:
                        self._latest = None
                        self._content_verified = False
                        self._last_observation_monotonic = None
                        self._freshness_error = None
                        self._last_error = f'窗口暂时没有可用画面，采集状态: {status}'
                return
            image_buffer = cm.CMSampleBufferGetImageBuffer(sample)
            if image_buffer is None:
                raise MacCaptureError('采集帧没有像素缓冲')
            if quartz.CVPixelBufferGetPixelFormatType(image_buffer) != quartz.kCVPixelFormatType_32BGRA:
                raise MacCaptureError('采集像素格式不是 BGRA')
            width = int(quartz.CVPixelBufferGetWidth(image_buffer))
            height = int(quartz.CVPixelBufferGetHeight(image_buffer))
            bytes_per_row = int(quartz.CVPixelBufferGetBytesPerRow(image_buffer))
            valid_rect, rect = quartz.CGRectMakeWithDictionaryRepresentation(
                metadata.get(sck.SCStreamFrameInfoContentRect), None,
            )
            if not valid_rect:
                raise MacCaptureError('采集帧缺少有效内容区')
            scale_factor = float(metadata[sck.SCStreamFrameInfoScaleFactor])
            content_scale_key = getattr(sck,'SCStreamFrameInfoContentScale',None)
            if (self._top_inset_points or self._auto_top_inset) and content_scale_key not in metadata:
                raise MacCaptureError('自动或逻辑点裁剪需要采集帧的内容缩放元数据')
            content_scale = float(metadata.get(content_scale_key,1.0))
            # 标题栏来自原窗口逻辑点，须包括内容缩放；contentRect本身已经包含该缩放。
            pixel_scale = scale_factor * content_scale
            if not math.isfinite(pixel_scale) or pixel_scale <= 0:
                raise MacCaptureError('采集帧的逻辑点与像素缩放无效')
            lock_flags = quartz.kCVPixelBufferLock_ReadOnly
            if quartz.CVPixelBufferLockBaseAddress(image_buffer, lock_flags) != 0:
                raise MacCaptureError('无法读取采集像素缓冲')
            try:
                address = quartz.CVPixelBufferGetBaseAddress(image_buffer)
                image = bgra_buffer_to_rgb(
                    address.as_buffer(height * bytes_per_row), width, height, bytes_per_row,
                    content_rect=(rect.origin.x, rect.origin.y, rect.size.width, rect.size.height),
                    scale_factor=scale_factor,
                )
            finally:
                quartz.CVPixelBufferUnlockBaseAddress(image_buffer, lock_flags)
            capture_time, monotonic_time = capture_timestamps(
                int(metadata[sck.SCStreamFrameInfoDisplayTime]),
                int(quartz.CVGetCurrentHostTime()),
                float(quartz.CVGetHostClockFrequency()),
                epoch_now=time.time(), monotonic_now=time.monotonic(),
            )
            image.flags.writeable = False
            with self._lock:
                if (not self._running or stream != self._stream
                        or activity_generation != self._activity_generation):
                    return
                # 跨 Space 的冻结画面也可能连续标为 Complete 且带新时间戳。
                # 首帧只作比较基准；后续只有像素变化才能授权输入。
                if self._requires_pixel_change:
                    previous = self._latest
                    if previous is not None and np.array_equal(previous.image, image):
                        return
                    self._content_verified = previous is not None
                else:
                    self._content_verified = True
                self._sequence += 1
                self._raw_size = (width, height)
                self._latest = CapturedFrame(image, self._sequence, capture_time, monotonic_time, width, height,pixel_scale)
                self._last_observation_monotonic = monotonic_time
                self._freshness_error = None
                self._last_error = None
        except Exception as error:
            self._fail_stream(stream, f'转换采集帧失败: {error}')

    def latest_raw_frame(self, *, max_age_seconds: float) -> CapturedFrame | None:
        """返回仍被有效采集观察确认的内容帧，保留其真实像素采集时间。"""
        if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError('允许的帧龄必须大于 0')
        with self._lock:
            frame = self._latest
            if not self._running or frame is None:
                return None
            if self._requires_pixel_change and not self._content_verified:
                # 初始化允许在原有帧龄期限内等第二张不同的图，不能把首帧
                # 尚未验证当成立即终止条件，也不能把它提供给输入层。
                if time.monotonic() - frame.monotonic_time > max_age_seconds:
                    self._freshness_error = '跨桌面尚未观察到画面变化，游戏可能已暂停渲染；拒绝使用冻结画面'
                return None
            observed_at = self._last_observation_monotonic
            age = time.monotonic() - max(frame.monotonic_time, observed_at or frame.monotonic_time)
            if not 0 <= age <= max_age_seconds:
                reason = '跨桌面画面未变化，游戏可能已暂停渲染' if self._requires_pixel_change else '窗口采集已过期'
                self._freshness_error = f'{reason}，最近有效观察距今 {age:.3f} 秒'
                return None
            return frame

    def latest_frame(self, *, max_age_seconds: float) -> CapturedFrame | None:
        """返回有实际采集时间的 1080p 帧，无帧、过期或比例不符时返回 None。"""
        with self._lock:
            stream = self._stream
            content_rect = self._content_rect
            top_inset_pixels = self._top_inset_pixels
            top_inset_points = self._top_inset_points
            auto_top_inset = self._auto_top_inset
        frame = self.latest_raw_frame(max_age_seconds=max_age_seconds)
        if frame is None:
            return None
        try:
            if top_inset_points:
                if not math.isfinite(frame.pixel_scale) or frame.pixel_scale <= 0:
                    raise ValueError('采集帧的逻辑点与像素缩放无效')
                top_inset_pixels = round(top_inset_points * frame.pixel_scale)
            image = normalize_game_frame(
                frame.image, content_rect=content_rect, top_inset_pixels=top_inset_pixels,
                auto_top_inset=auto_top_inset, pixel_scale=frame.pixel_scale,
            )
        except ValueError as error:
            with self._lock:
                if self._running and self._stream == stream:
                    self._last_error = str(error)
            return None
        with self._lock:
            if not self._running or self._stream != stream:
                return None
        return replace(frame, image=image)

    def stop(self) -> None:
        """先清空帧和绑定，再停止原生采集，丢弃所有迟到回调。"""
        with self._audio_control_lock:
            self._stop_stream()

    def _stop_stream(self) -> None:
        """与声音订阅串行停止，防止关闭窗口时重新挂入声音输出。"""
        with self._lock:
            stream = self._stream
        self._fail_audio(stream, '游戏窗口采集已停止，已清空声音缓冲')
        self.stop_audio()
        with self._lock:
            stream, output = self._stream, self._output
            self._stream = self._output = self._queue = None
            self._configuration = None
            self._running = False
            self._active = False
            self._requires_pixel_change = False
            self._content_verified = False
            self._last_sample_monotonic = None
            self._sample_status = None
            self._sample_counts.clear()
            self._latest = None
            self._last_observation_monotonic = None
            self._freshness_error = None
            self._raw_size = None
            self.window = None
            self._content_rect = None
            self._top_inset_pixels = 0
            self._top_inset_points = 0
            self._auto_top_inset = False
        if output is not None:
            output.bind(None)
        if stream is not None:
            try:
                error, = self._wait(stream.stopCaptureWithCompletionHandler_, '停止窗口采集')
                if error is not None:
                    raise MacCaptureError(f'停止窗口采集失败: {error}')
            except Exception as error:
                with self._lock:
                    self._last_error = str(error)
