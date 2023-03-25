from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *

import ctypes
import _ctypes
from _ctypes import COMError
import comtypes
import sys
import numpy
import time
import threading
from numpy.lib import recfunctions as rfn

KINECT_MAX_BODY_COUNT = 6
TIME_GAP = 1679157165


class PyKinectRuntime(object):
    """manages Kinect objects and simplifying access to them"""

    def __init__(self, frame_source_types):
        # recipe to get address of surface: http://archives.seul.org/pygame/users/Apr-2008/msg00218.html
        is_64bits = sys.maxsize > 2 ** 32
        if not is_64bits:
            self.Py_ssize_t = ctypes.c_int
        else:
            self.Py_ssize_t = ctypes.c_int64

        self._PyObject_AsWriteBuffer = ctypes.pythonapi.PyObject_AsWriteBuffer
        self._PyObject_AsWriteBuffer.restype = ctypes.c_int
        self._PyObject_AsWriteBuffer.argtypes = [ctypes.py_object,
                                                 ctypes.POINTER(ctypes.c_void_p),
                                                 ctypes.POINTER(self.Py_ssize_t)]

        self._close_event = ctypes.windll.kernel32.CreateEventW(None, False, False, None)

        self._color_frame_arrived_event = 0
        self._depth_frame_arrived_event = 0
        self._body_frame_arrived_event = 0
        self._body_index_frame_arrived_event = 0
        self._infrared_frame_arrived_event = 0
        self._long_exposure_infrared_frame_arrived_event = 0
        self._audio_frame_arrived_event = 0

        self._color_frame_lock = threading.Lock()
        self._depth_frame_lock = threading.Lock()
        self._body_frame_lock = threading.Lock()
        self._body_index_frame_lock = threading.Lock()
        self._infrared_frame_lock = threading.Lock()
        self._long_exposure_infrared_frame_lock = threading.Lock()
        self._audio_frame_lock = threading.Lock()

        # initialize sensor
        self._sensor = ctypes.POINTER(PyKinectV2.IKinectSensor)()
        hres = ctypes.windll.kinect20.GetDefaultKinectSensor(ctypes.byref(self._sensor))
        hres = self._sensor.Open()

        self._mapper = self._sensor.CoordinateMapper

        self.frame_source_types = frame_source_types
        self.max_body_count = KINECT_MAX_BODY_COUNT

        self._handles = (ctypes.c_voidp * 8)()
        self._handles[0] = self._close_event
        self._handles[1] = self._close_event
        self._handles[2] = self._close_event
        self._handles[3] = self._close_event
        self._handles[4] = self._close_event
        self._handles[5] = self._close_event
        self._handles[6] = self._close_event
        self._handles[7] = self._close_event

        self._waitHandleCount = 1

        self._color_source = self._sensor.ColorFrameSource
        self.color_frame_desc = self._color_source.FrameDescription
        self._infrared_source = self._sensor.InfraredFrameSource
        self.infrared_frame_desc = self._infrared_source.FrameDescription
        self._depth_source = self._sensor.DepthFrameSource
        self.depth_frame_desc = self._depth_source.FrameDescription
        self._body_index_source = self._sensor.BodyIndexFrameSource
        self.body_index_frame_desc = self._body_index_source.FrameDescription
        self._body_source = self._sensor.BodyFrameSource
        self.max_body_count = self._body_source.BodyCount

        self._color_frame_data = None
        self._depth_frame_data = None
        self._body_frame_data = None
        self._body_index_frame_data = None
        self._infrared_frame_data = None
        self._long_exposure_infrared_frame_data = None
        self._audio_frame_data = None

        if self.frame_source_types & FrameSourceTypes_Color:
            self._color_frame_data = ctypes.POINTER(ctypes.c_ubyte)
            self._color_frame_data_capacity = ctypes.c_uint(
                self.color_frame_desc.Width * self.color_frame_desc.Height * 4)
            self._color_frame_data_type = ctypes.c_ubyte * self._color_frame_data_capacity.value
            self._color_frame_data = ctypes.cast(self._color_frame_data_type(), ctypes.POINTER(ctypes.c_ubyte))
            self._color_frame_reader = self._color_source.OpenReader()
            self._color_frame_arrived_event = self._color_frame_reader.SubscribeFrameArrived()
            self._handles[self._waitHandleCount] = self._color_frame_arrived_event
            self._waitHandleCount += 1

        if self.frame_source_types & FrameSourceTypes_Infrared:
            self._infrared_frame_data = ctypes.POINTER(ctypes.c_ushort)
            self._infrared_frame_data_capacity = ctypes.c_uint(
                self.infrared_frame_desc.Width * self.infrared_frame_desc.Height)
            self._infrared_frame_data_type = ctypes.c_ushort * self._infrared_frame_data_capacity.value
            self._infrared_frame_data = ctypes.cast(self._infrared_frame_data_type(), ctypes.POINTER(ctypes.c_ushort))
            self._infrared_frame_reader = self._infrared_source.OpenReader()
            self._infrared_frame_arrived_event = self._infrared_frame_reader.SubscribeFrameArrived()
            self._handles[self._waitHandleCount] = self._infrared_frame_arrived_event
            self._waitHandleCount += 1

        if self.frame_source_types & FrameSourceTypes_Depth:
            self._depth_frame_data = ctypes.POINTER(ctypes.c_ushort)
            self._depth_frame_data_capacity = ctypes.c_uint(self.depth_frame_desc.Width * self.depth_frame_desc.Height)
            self._depth_frame_data_type = ctypes.c_ushort * self._depth_frame_data_capacity.value
            self._depth_frame_data = ctypes.cast(self._depth_frame_data_type(), ctypes.POINTER(ctypes.c_ushort))
            self._depth_frame_reader = self._depth_source.OpenReader()
            self._depth_frame_arrived_event = self._depth_frame_reader.SubscribeFrameArrived()
            self._handles[self._waitHandleCount] = self._depth_frame_arrived_event
            self._waitHandleCount += 1

        if self.frame_source_types & FrameSourceTypes_BodyIndex:
            self._body_index_frame_data = ctypes.POINTER(ctypes.c_ubyte)
            self._body_index_frame_data_capacity = ctypes.c_uint(
                self.body_index_frame_desc.Width * self.body_index_frame_desc.Height)
            self._body_index_frame_data_type = ctypes.c_ubyte * self._body_index_frame_data_capacity.value
            self._body_index_frame_data = ctypes.cast(self._body_index_frame_data_type(),
                                                      ctypes.POINTER(ctypes.c_ubyte))
            self._body_index_frame_reader = self._body_index_source.OpenReader()
            self._body_index_frame_arrived_event = self._body_index_frame_reader.SubscribeFrameArrived()
            self._handles[self._waitHandleCount] = self._body_index_frame_arrived_event
            self._waitHandleCount += 1

        if self.frame_source_types & FrameSourceTypes_Body:
            self._body_frame_data_capacity = ctypes.c_uint(self.max_body_count)
            self._body_frame_data_type = ctypes.POINTER(IBody) * self._body_frame_data_capacity.value
            self._body_frame_data = ctypes.cast(self._body_frame_data_type(), ctypes.POINTER(ctypes.POINTER(IBody)))
            self._body_frame_reader = self._body_source.OpenReader()
            self._body_frame_arrived_event = self._body_frame_reader.SubscribeFrameArrived()
            self._body_frame_bodies = None
            self._handles[self._waitHandleCount] = self._body_frame_arrived_event
            self._waitHandleCount += 1

        # argument of target cannot be enclosed in parentheses
        new_thread = threading.Thread(target=self.kinect_frame_thread, args=(), daemon=True)
        new_thread.start()
        # thread.start_new_thread(self.kinect_frame_thread, ())

        self._last_color_frame = None
        self._last_depth_frame = None
        self._last_body_frame = None
        self._last_body_index_frame = None
        self._last_infrared_frame = None
        self._last_long_exposure_infrared_frame = None
        self._last_audio_frame = None

        start_clock = time.perf_counter()
        self._last_color_frame_access = self._last_color_frame_arrival = start_clock
        self._last_body_frame_access = self._last_body_frame_arrival = start_clock
        self._last_body_index_frame_access = self._last_body_index_frame_arrival = start_clock
        self._last_depth_frame_access = self._last_depth_frame_arrival = start_clock
        self._last_infrared_frame_access = self._last_infrared_frame_arrival = start_clock
        self._last_long_exposure_infrared_frame_access = self._last_long_exposure_infrared_frame_arrival = start_clock
        self._last_audio_frame_access = self._last_audio_frame_arrival = start_clock

        self._last_color_frame_time = None
        self._last_body_frame_time = None
        self._last_body_index_frame_time = None
        self._last_depth_frame_time = None
        self._last_infrared_frame_time = None
        self._last_long_exposure_infrared_frame_time = None
        self._last_audio_frame_time = None

        self._first_time = True

    @property
    def first_time(self):
        return self._first_time

    @first_time.setter
    def first_time(self, value):
        self._first_time = value

    def close(self):
        if self._sensor is not None:
            ctypes.windll.kernel32.SetEvent(self._close_event)
            ctypes.windll.kernel32.CloseHandle(self._close_event)

            self._color_frame_reader = None
            self._depth_frame_reader = None
            self._body_index_frame_reader = None
            self._body_frame_reader = None

            self._color_source = None
            self._depth_source = None
            self._body_index_source = None
            self._body_source = None

            self._body_frame_data = None

            self._sensor.Close()
            self._sensor = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def surface_as_array(self, surface_buffer_interface):
        address = ctypes.c_void_p()
        size = self.Py_ssize_t()
        self._PyObject_AsWriteBuffer(surface_buffer_interface,
                                     ctypes.byref(address), ctypes.byref(size))
        bytes = (ctypes.c_byte * size.value).from_address(address.value)
        bytes.object = surface_buffer_interface
        return bytes

    def kinect_frame_thread(self):
        while True:
            wait = ctypes.windll.kernel32.WaitForMultipleObjects(self._waitHandleCount, self._handles, False,
                                                                 PyKinectV2._INFINITE)
            if wait == 0:
                break
            if self._handles[wait] == self._color_frame_arrived_event:
                self.handle_color_arrived(wait)
            elif self._handles[wait] == self._depth_frame_arrived_event:
                self.handle_depth_arrived(wait)
            elif self._handles[wait] == self._body_frame_arrived_event:
                self.handle_body_arrived(wait)
            elif self._handles[wait] == self._body_index_frame_arrived_event:
                self.handle_body_index_arrived(wait)
            elif self._handles[wait] == self._infrared_frame_arrived_event:
                self.handle_infrared_arrived(wait)
            elif self._handles[wait] == self._long_exposure_infrared_frame_arrived_event:
                self.handle_long_exposure_infrared_arrived(wait)
            elif self._handles[wait] == self._audio_frame_arrived_event:
                self.handle_audio_arrived(wait)
            else:
                break

    """
    color frame related functions:
     - handle_color_arrived
     - has_new_color_frame
     - get_last_color_frame_data
     - get_last_color_frame_time
     - get_last_color_frame_infor
    """

    def handle_color_arrived(self, handle_index):
        color_frame_event_data = self._color_frame_reader.GetFrameArrivedEventData(self._handles[handle_index])
        color_frame_ref = color_frame_event_data.FrameReference
        try:
            color_frame = color_frame_ref.AcquireFrame()
            try:
                with self._color_frame_lock:
                    color_frame.CopyConvertedFrameDataToArray(self._color_frame_data_capacity, self._color_frame_data,
                                                              PyKinectV2.ColorImageFormat_Bgra)
                    self._last_color_frame_arrival = time.perf_counter()
                    self._last_color_frame_time = color_frame.RelativeTime / 1e7 + TIME_GAP
            except:
                pass
            color_frame = None
        except:
            pass
        color_frame_ref = None
        color_frame_event_data = None

    def has_new_color_frame(self):
        has = (self._last_color_frame_arrival > self._last_color_frame_access)
        return has

    def get_last_color_frame_data(self):
        with self._color_frame_lock:
            if self._color_frame_data is not None:
                data = numpy.copy(numpy.ctypeslib.as_array(self._color_frame_data,
                                                           shape=(self._color_frame_data_capacity.value,)))
                self._last_color_frame_access = time.perf_counter()
                return data
            else:
                return None

    def get_last_color_frame_time(self):
        return self._last_color_frame_time

    def get_last_color_frame_infor(self):
        color_frame_image = None
        color_frame_time = None
        if self.has_new_color_frame():
            data = self.get_last_color_frame_data()
            color_frame_time = self.get_last_color_frame_time()
            reshape_data = data.reshape([self.color_frame_desc.Height,
                                         self.color_frame_desc.Width, 4])
            color_frame_image = reshape_data[:, :, 0:3]
        return color_frame_image, color_frame_time

    """
    infrared frame related functions:
     - handle_infrared_arrived
     - has_new_infrared_frame
     - get_last_infrared_frame_data
     - get_last_infrared_frame_time
     - get_last_infrared_frame_time
    """

    def handle_infrared_arrived(self, handle_index):
        infrared_frame_event_data = self._infrared_frame_reader.GetFrameArrivedEventData(self._handles[handle_index])
        infrared_frame_ref = infrared_frame_event_data.FrameReference
        try:
            infrared_frame = infrared_frame_ref.AcquireFrame()
            try:
                with self._infrared_frame_lock:
                    infrared_frame.CopyFrameDataToArray(self._infrared_frame_data_capacity, self._infrared_frame_data)
                    self._last_infrared_frame_arrival = time.perf_counter()
                    self._last_infrared_frame_time = infrared_frame.RelativeTime / 1e7 + TIME_GAP
            except:
                pass
            infrared_frame = None
        except:
            pass
        infrared_frame_ref = None
        infrared_frame_event_data = None

    def has_new_infrared_frame(self):
        has = (self._last_infrared_frame_arrival > self._last_infrared_frame_access)
        return has

    def get_last_infrared_frame_data(self):
        with self._infrared_frame_lock:
            if self._infrared_frame_data is not None:
                data = numpy.copy(numpy.ctypeslib.as_array(self._infrared_frame_data,
                                                           shape=(self._infrared_frame_data_capacity.value,)))
                self._last_infrared_frame_access = time.perf_counter()
                return data
            else:
                return None

    def get_last_infrared_frame_time(self):
        return self._last_infrared_frame_time

    def get_last_infrared_frame_infor(self):
        infrared_frame_image = None
        infrared_frame_time = None
        if self.has_new_infrared_frame():
            data = self.get_last_infrared_frame_data()
            infrared_frame_time = self.get_last_infrared_frame_time()
            reshape_data = data.reshape([self.infrared_frame_desc.Height,
                                         self.infrared_frame_desc.Width, 1])
            infrared_frame_image = reshape_data[:, :, :]
        return infrared_frame_image, infrared_frame_time

    """
    depth frame related functions:
     - handle_depth_arrived
     - has_new_depth_frame
     - get_last_depth_frame_data
     - get_last_depth_frame_time
     - get_last_depth_frame_infor
    """

    def handle_depth_arrived(self, handle_index):
        depth_frame_event_data = self._depth_frame_reader.GetFrameArrivedEventData(self._handles[handle_index])
        depth_frame_ref = depth_frame_event_data.FrameReference
        try:
            depth_frame = depth_frame_ref.AcquireFrame()
            try:
                with self._depth_frame_lock:
                    depth_frame.CopyFrameDataToArray(self._depth_frame_data_capacity, self._depth_frame_data)
                    self._last_depth_frame_arrival = time.perf_counter()
                    self._last_depth_frame_time = depth_frame.RelativeTime / 1e7 + TIME_GAP
            except:
                pass
            depth_frame = None
        except:
            pass
        depth_frame_ref = None
        depth_frame_event_data = None

    def has_new_depth_frame(self):
        has = (self._last_depth_frame_arrival > self._last_depth_frame_access)
        return has

    def get_last_depth_frame_data(self):
        with self._depth_frame_lock:
            if self._depth_frame_data is not None:
                data = numpy.copy(numpy.ctypeslib.as_array(self._depth_frame_data,
                                                           shape=(self._depth_frame_data_capacity.value,)))
                self._last_depth_frame_access = time.perf_counter()
                return data
            else:
                return None

    def get_last_depth_frame_time(self):
        return self._last_depth_frame_time

    def get_last_depth_frame_infor(self):
        depth_frame_image = None
        depth_frame_time = None
        if self.has_new_depth_frame():
            data = self.get_last_depth_frame_data()
            depth_frame_time = self.get_last_depth_frame_time()
            reshape_data = data.reshape([self.depth_frame_desc.Height,
                                         self.depth_frame_desc.Width, 1])
            depth_frame_image = reshape_data[:, :, :]
        return depth_frame_image, depth_frame_time

    """
    body index frame related functions:
     - handle_body_index_arrived
     - has_new_body_index_frame
     - get_last_body_index_frame_data
     - get_last_body_index_frame_time
     - get_last_body_index_frame_infor (unfinished)
    """

    def handle_body_index_arrived(self, handle_index):
        body_index_frame_event_data = self._body_index_frame_reader.GetFrameArrivedEventData(
            self._handles[handle_index])
        body_index_frame_ref = body_index_frame_event_data.FrameReference
        try:
            body_index_frame = body_index_frame_ref.AcquireFrame()
            try:
                with self._body_index_frame_lock:
                    body_index_frame.CopyFrameDataToArray(self._body_index_frame_data_capacity,
                                                          self._body_index_frame_data)
                    self._last_body_index_frame_arrival = time.perf_counter()
                    self._last_body_index_frame_time = body_index_frame.RelativeTime / 1e7 + TIME_GAP
            except:
                pass
            body_index_frame = None
        except:
            pass
        body_index_frame = None
        body_index_frame_event_data = None

    def has_new_body_index_frame(self):
        has = (self._last_body_index_frame_arrival > self._last_body_index_frame_access)
        return has

    def get_last_body_index_frame_data(self):
        with self._body_index_frame_lock:
            if self._body_index_frame_data is not None:
                data = numpy.copy(numpy.ctypeslib.as_array(self._body_index_frame_data,
                                                           shape=(self._body_index_frame_data_capacity.value,)))
                self._last_body_index_frame_access = time.perf_counter()
                return data
            else:
                return None

    def get_last_body_index_frame_time(self):
        return self._last_body_index_frame_time

    def get_last_body_index_frame_infor(self):
        pass

    """
    body frame related functions:
     - handle_body_arrived
     - has_new_body_frame
     - get_last_body_frame_data
     - get_last_body_frame_time
     - get_last_body_frame_infor (unfinished)
    """

    def handle_body_arrived(self, handle_index):
        body_frame_event_data = self._body_frame_reader.GetFrameArrivedEventData(self._handles[handle_index])
        body_frame_ref = body_frame_event_data.FrameReference
        try:
            body_frame = body_frame_ref.AcquireFrame()
            try:
                with self._body_frame_lock:
                    body_frame.GetAndRefreshBodyData(self._body_frame_data_capacity, self._body_frame_data)
                    self._body_frame_bodies = KinectBodyFrameData(body_frame, self._body_frame_data,
                                                                  self.max_body_count)
                    self._last_body_frame_arrival = time.perf_counter()
                    self._last_body_frame_time = body_frame.RelativeTime / 1e7 + TIME_GAP

                # need these 2 lines as a workaround for handling IBody referencing exception
                self._body_frame_data = None
                self._body_frame_data = ctypes.cast(self._body_frame_data_type(), ctypes.POINTER(ctypes.POINTER(IBody)))
            except:
                pass
            body_frame = None
        except:
            pass
        body_frame_ref = None
        body_frame_event_data = None

    def has_new_body_frame(self):
        has = (self._last_body_frame_arrival > self._last_body_frame_access)
        return has

    def get_last_body_frame_data(self):
        with self._body_frame_lock:
            if self._body_frame_bodies is not None:
                self._last_body_frame_access = time.perf_counter()
                return self._body_frame_bodies.copy()
            else:
                return None

    def get_last_body_frame_time(self):
        return self._last_body_frame_time

    def get_last_body_frame_infor(self):
        pass

    """
    long exposure infrared frame related functions:
     - handle_long_exposure_infrared_arrived (unfinished)
     - has_new_long_exposure_infrared_frame
     - get_last_long_exposure_infrared_frame_data (unfinished)
     - get_last_long_exposure_infrared_frame_time
     - get_last_long_exposure_infrared_frame_infor (unfinished)
    """

    def handle_long_exposure_infrared_arrived(self, handle_index):
        pass

    def has_new_long_exposure_infrared_frame(self):
        has = (self._last_long_exposure_infrared_frame_arrival > self._last_long_exposure_infrared_frame_access)
        return has

    def get_last_long_exposure_infrared_frame_data(self):
        pass

    def get_last_long_exposure_infrared_frame_time(self):
        return self._last_long_exposure_infrared_frame_time

    def get_last_long_exposure_infrared_frame_infor(self):
        pass

    """
    audio frame related functions:
     - handle_audio_arrived (unfinished)
     - has_new_audio_frame
     - get_last_audio_frame_data (unfinished)
     - get_last_audio_frame_time
     - get_last_audio_frame_infor (unfinished)
    """

    def handle_audio_arrived(self, handle_index):
        pass

    def has_new_audio_frame(self):
        has = (self._last_audio_frame_arrival > self._last_audio_frame_access)
        return has

    def get_last_audio_frame_data(self):
        pass

    def get_last_audio_frame_time(self):
        return self._last_audio_frame_time

    def get_last_audio_frame_infor(self):
        pass

    """
    coordinate mapper related functions
    """

    def body_joint_to_color_space(self, joint):
        return self._mapper.MapCameraPointToColorSpace(joint.Position)

    def body_joint_to_depth_space(self, joint):
        return self._mapper.MapCameraPointToDepthSpace(joint.Position)

    def body_joints_to_color_space(self, joints):
        joint_points = numpy.ndarray((PyKinectV2.JointType_Count), dtype=numpy.object)

        for j in range(0, PyKinectV2.JointType_Count):
            joint_points[j] = self.body_joint_to_color_space(joints[j])

        return joint_points

    def body_joints_to_depth_space(self, joints):
        joint_points = numpy.ndarray((PyKinectV2.JointType_Count), dtype=numpy.object)

        for j in range(0, PyKinectV2.JointType_Count):
            joint_points[j] = self.body_joint_to_depth_space(joints[j])

        return joint_points

    def map_depth_to_color(self):
        depth2color_points_type = _DepthSpacePoint * int(512 * 424)
        depth2color_points = ctypes.cast(depth2color_points_type(), ctypes.POINTER(_ColorSpacePoint))
        self._mapper.MapDepthFrameToColorSpace(ctypes.c_uint(512 * 424), self._depth_frame_data,
                                               ctypes.c_uint(512 * 424), depth2color_points)
        # reference: https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
        color_xy_structured = numpy.copy(numpy.ctypeslib.as_array(depth2color_points, shape=(424 * 512,)))
        color_xy_unstructured = rfn.structured_to_unstructured(color_xy_structured)
        color_xy = color_xy_unstructured.reshape((424, 512, 2)).astype(int)
        color_x = numpy.clip(color_xy[:, :, 0], 0, 1920 - 1)
        color_y = numpy.clip(color_xy[:, :, 1], 0, 1080 - 1)
        # color_x = color_xy[:, :, 0]
        # color_y = color_xy[:, :, 1]
        return color_x, color_y

    def map_color_to_depth(self):
        color2depth_points_type = _ColorSpacePoint * int(1920 * 1080)
        color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(_DepthSpacePoint))
        self._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), self._depth_frame_data,
                                               ctypes.c_uint(1920 * 1080), color2depth_points)
        depth_xy_structured = numpy.copy(numpy.ctypeslib.as_array(color2depth_points, shape=(1080 * 1920,)))
        depth_xy_unstructured = rfn.structured_to_unstructured(depth_xy_structured)
        depth_xy = depth_xy_unstructured.reshape((1080, 1920, 2)).astype(int)
        depth_x = numpy.clip(depth_xy[:, :, 0], 0, 512 - 1)
        depth_y = numpy.clip(depth_xy[:, :, 1], 0, 424 - 1)
        # depth_x = depth_xy[:, :, 0]
        # depth_y = depth_xy[:, :, 1]
        return depth_x, depth_y


class KinectBody(object):
    def __init__(self, body=None):
        self.is_restricted = False
        self.tracking_id = -1

        self.is_tracked = False

        if body is not None:
            self.is_tracked = body.IsTracked

        if self.is_tracked:
            self.is_restricted = body.IsRestricted
            self.tracking_id = body.TrackingId
            self.engaged = body.Engaged
            self.lean = body.Lean
            self.lean_tracking_state = body.LeanTrackingState
            self.hand_left_state = body.HandLeftState
            self.hand_left_confidence = body.HandLeftConfidence
            self.hand_right_state = body.HandRightState
            self.hand_right_confidence = body.HandRightConfidence
            self.clipped_edges = body.ClippedEdges

            joints = ctypes.POINTER(PyKinectV2._Joint)
            joints_capacity = ctypes.c_uint(PyKinectV2.JointType_Count)
            joints_data_type = PyKinectV2._Joint * joints_capacity.value
            joints = ctypes.cast(joints_data_type(), ctypes.POINTER(PyKinectV2._Joint))
            body.GetJoints(PyKinectV2.JointType_Count, joints)
            self.joints = joints

            joint_orientations = ctypes.POINTER(PyKinectV2._JointOrientation)
            joint_orientations_data_type = PyKinectV2._JointOrientation * joints_capacity.value
            joint_orientations = ctypes.cast(joint_orientations_data_type(),
                                             ctypes.POINTER(PyKinectV2._JointOrientation))
            body.GetJointOrientations(PyKinectV2.JointType_Count, joint_orientations)
            self.joint_orientations = joint_orientations


class KinectBodyFrameData(object):
    def __init__(self, bodyFrame, body_frame_data, max_body_count):
        self.bodies = None
        self.floor_clip_plane = None
        if bodyFrame is not None:
            self.floor_clip_plane = bodyFrame.FloorClipPlane
            self.relative_time = bodyFrame.RelativeTime

            self.bodies = numpy.ndarray((max_body_count), dtype=numpy.object)
            for i in range(0, max_body_count):
                self.bodies[i] = KinectBody(body_frame_data[i])

    def copy(self):
        res = KinectBodyFrameData(None, None, 0)
        res.floor_clip_plane = self.floor_clip_plane
        res.relative_time = self.relative_time
        res.bodies = numpy.copy(self.bodies)
        return res
