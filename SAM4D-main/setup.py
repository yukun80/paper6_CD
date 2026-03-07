import os
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension, CUDAExtension)


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):
    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        if module == 'det3d.ops.point_deep':
            sources = sources_cuda
        else:
            sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        raise EnvironmentError('CUDA is required to compile!')

    print(f'{name}, extra_include_path: {extra_include_path}')
    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    setup(
        name='SAM4D',
        version='v1.0',
        description=("SAM4D: Segment Anything in Camera and LiDAR Streams"),
        packages=find_packages(exclude=['data', 'notebooks']),
        include_package_data=True,
        package_data={'sam4d.ops': ['*/*.so']},
        ext_modules=[
            make_cuda_ext(
                name='cc_ext',
                module='sam4d.ops.sam2',
                sources=[],
                sources_cuda=['csrc/connected_components.cu']),

        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
    )
