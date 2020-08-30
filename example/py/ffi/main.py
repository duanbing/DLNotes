# filename: main.py
import ctypes

class PyObj(ctypes.Structure):
    _fields_ = [("f", ctypes.c_int), ("f2", ctypes.c_int)]

class PyObjRet (ctypes.Structure):
    _fields_ = [("o", ctypes.c_int), ("r", ctypes.c_int)]

so = ctypes.CDLL("./libadd.so")
so.add_obj.restype = PyObjRet;
so.AddObjCallback.restype = PyObjRet;

num = so.add(7, 9)
print(num)

obj = PyObj(10,90);
print(obj.f)
ret = so.add_obj(ctypes.byref(obj));
print(type(ret), ret.o, ret.r)

def add_obj_callback(a, b):
    print(type(a[0]), type(b[0]), a.contents)
    # https://docs.python.org/2/library/ctypes.html#ctypes._Pointer.contents
    a = a.contents
    b = b.contents
    b.o = a.f + a.f2
    b.r = a.f;

rettype = ctypes.CFUNCTYPE(None, ctypes.POINTER(PyObj), ctypes.POINTER(PyObjRet));
obj = PyObj(101, 90);
print(obj.f)
#ret = so.AddObjCallback(rettype(add_obj_callback), ctypes.byref(obj));
ret = so.AddObjCallback(rettype(add_obj_callback), (obj));
print(type(ret), "sum = ", ret.o)


