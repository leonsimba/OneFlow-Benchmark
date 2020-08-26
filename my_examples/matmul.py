# python3 matmul.py
import oneflow as flow
import oneflow.typing as tp
import numpy as np

flow.config.enable_debug_mode(True)
#flow.config.gpu_device_num(1)
flow.config.gpu_device_num(2)
#flow.env.ctrl_port(10022)
#nodes = [{"addr": "10.38.30.82"}, {"addr": "10.38.30.77"}]
#flow.env.machine(nodes)

@flow.global_function()
def Matmul(
    x: tp.Numpy.Placeholder((4, 4), dtype=flow.float32, batch_axis=None),
    y: tp.Numpy.Placeholder((4, 4), dtype=flow.float32, batch_axis=1)) -> tp.Numpy:

    print("---------------------------")
    print("x.split_axis:", x.split_axis)
    print("x.parallel_conf.device_tag:" , x.parallel_conf.device_tag)
    print("x.parallel_conf.device_name:", x.parallel_conf.device_name)

    print("---------------------------")
    print("y.split_axis:", y.split_axis)
    print("y.parallel_conf.device_tag:" , y.parallel_conf.device_tag)
    print("y.parallel_conf.device_name:", y.parallel_conf.device_name)

    s = flow.matmul(x, y)

    print("---------------------------")
    print("w.split_axis:", s.split_axis)
    print("w.parallel_conf.device_tag:" , s.parallel_conf.device_tag)
    print("w.parallel_conf.device_name:", s.parallel_conf.device_name)

    z = flow.matmul(s, x)

    print("---------------------------")
    print("z.split_axis:", z.split_axis)
    print("z.parallel_conf.device_tag:" , z.parallel_conf.device_tag)
    print("z.parallel_conf.device_name:", z.parallel_conf.device_name)

    return z

x = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
]).astype(np.float32) 

y = np.array([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4, 4, 4],
]).astype(np.float32) 

print("----- oneflow -----")
print(Matmul(x, y))
print("----- numpy -----")
print(np.matmul(np.matmul(x, y), x))
