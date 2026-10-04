"""Read NvDsOpticalFlowMeta from GStreamer buffers (no pyds required)."""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER, Structure, c_float, c_int, c_uint, c_ulong, c_void_p, c_short,
)

import numpy as np

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except (ImportError, ValueError):
    Gst = None

NVDS_OPTICAL_FLOW_META = 10  # NvDsMetaType::NVDS_OPTICAL_FLOW_META


class GList(Structure):
    pass


GList._fields_ = [
    ("data", c_void_p),
    ("next", POINTER(GList)),
    ("prev", POINTER(GList)),
]


class NvDsBaseMeta(Structure):
    _fields_ = [
        ("batch_meta", c_void_p),
        ("meta_type", c_int),
        ("uContext", c_void_p),
        ("copy_func", c_void_p),
        ("release_func", c_void_p),
    ]


class NvDsUserMeta(Structure):
    _fields_ = [
        ("base_meta", NvDsBaseMeta),
        ("user_meta_data", c_void_p),
    ]


class NvDsOpticalFlowMeta(Structure):
    _fields_ = [
        ("rows", c_uint),
        ("cols", c_uint),
        ("mv_size", c_uint),
        ("cost_size", c_uint),
        ("frame_num", c_ulong),
        ("data", c_void_p),
        ("cost", c_void_p),
        ("priv", c_void_p),
        ("reserved", c_void_p),
    ]


class NvDsFrameMeta(Structure):
    _fields_ = [
        ("base_meta", NvDsBaseMeta),
        ("pad_index", c_uint),
        ("batch_id", c_uint),
        ("frame_num", c_int),
        ("buf_pts", c_ulong),
        ("ntp_timestamp", c_ulong),
        ("source_id", c_uint),
        ("num_surfaces_per_frame", c_int),
        ("source_frame_width", c_uint),
        ("source_frame_height", c_uint),
        ("surface_type", c_uint),
        ("surface_index", c_uint),
        ("num_obj_meta", c_uint),
        ("bInferDone", c_int),
        ("obj_meta_list", POINTER(GList)),
        ("display_meta_list", POINTER(GList)),
        ("frame_user_meta_list", POINTER(GList)),
    ]


class NvDsBatchMeta(Structure):
    _fields_ = [
        ("base_meta", NvDsBaseMeta),
        ("max_frames_in_batch", c_uint),
        ("num_frames_in_batch", c_uint),
        ("frame_meta_pool", c_void_p),
        ("obj_meta_pool", c_void_p),
        ("classifier_meta_pool", c_void_p),
        ("display_meta_pool", c_void_p),
        ("user_meta_pool", c_void_p),
        ("label_info_meta_pool", c_void_p),
        ("frame_meta_list", POINTER(GList)),
        ("batch_user_meta_list", POINTER(GList)),
        ("meta_mutex", c_void_p * 8),
        ("misc_batch_info", c_ulong * 4),
        ("reserved", c_ulong * 4),
    ]


_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    for name in (
        "nvdsgst_meta",
        "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvdsgst_meta.so",
    ):
        try:
            _lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if _lib is None:
        raise OSError("libnvdsgst_meta.so not found")
    _lib.gst_buffer_get_nvds_batch_meta.argtypes = [c_void_p]
    _lib.gst_buffer_get_nvds_batch_meta.restype = c_void_p
    return _lib


def _buffer_ptr(gst_buffer) -> int:
    if hasattr(gst_buffer, "__gpointer__"):
        return int(gst_buffer.__gpointer__)
    return hash(gst_buffer)


def _iter_glist(head: POINTER(GList) | None):
    node = head
    while node:
        yield node.contents.data
        node = node.contents.next


def extract_flow_from_buffer(gst_buffer) -> np.ndarray | None:
    """Return flow field (rows, cols, 2) float32 in pixel units, or None."""
    if gst_buffer is None:
        return None
    lib = _load_lib()
    batch_ptr = lib.gst_buffer_get_nvds_batch_meta(_buffer_ptr(gst_buffer))
    if not batch_ptr:
        return None
    batch = NvDsBatchMeta.from_address(batch_ptr)
    for frame_ptr in _iter_glist(batch.frame_meta_list):
        if not frame_ptr:
            continue
        frame = NvDsFrameMeta.from_address(frame_ptr)
        for user_ptr in _iter_glist(frame.frame_user_meta_list):
            if not user_ptr:
                continue
            user = NvDsUserMeta.from_address(user_ptr)
            if user.base_meta.meta_type != NVDS_OPTICAL_FLOW_META:
                continue
            if not user.user_meta_data:
                continue
            ofmeta = NvDsOpticalFlowMeta.from_address(user.user_meta_data)
            rows, cols = int(ofmeta.rows), int(ofmeta.cols)
            if rows <= 0 or cols <= 0 or not ofmeta.data:
                continue
            count = rows * cols
            raw = (c_short * (count * 2)).from_address(ofmeta.data)
            arr = np.frombuffer(raw, dtype=np.int16).reshape(rows, cols, 2)
            return arr.astype(np.float32)
    return None
