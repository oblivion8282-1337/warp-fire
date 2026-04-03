from setuptools import setup, Extension
import subprocess

def pkg_config(lib, what):
    return subprocess.check_output(
        ["pkg-config", what, lib], text=True
    ).strip().split()

sdl2_cflags = pkg_config("sdl2", "--cflags")
sdl2_libs = pkg_config("sdl2", "--libs")

ext = Extension(
    "vk_interop",
    sources=["vk_interop.c"],
    extra_compile_args=[*sdl2_cflags, "-I/opt/cuda/include"],
    extra_link_args=[*sdl2_libs, "-lvulkan", "-L/opt/cuda/lib64", "-lcudart"],
)

setup(
    name="vk_interop",
    ext_modules=[ext],
)
