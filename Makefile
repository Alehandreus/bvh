build_cpu:
	g++ \
	examples/example_cpu.cpp \
	src/utils.cpp \
	src/build.cpp \
	src/cpu_traverse.cpp \
	bvh/src/bvh/v2/c_api/bvh.cpp \
	-Ibvh/src/ \
	-lassimp \
	-fopenmp \
	-std=c++20 \
	-O3 \
	-o bvh.out

build_gpu:
	nvcc \
	examples/example_gpu.cu \
	src/utils.cpp \
	src/build.cpp \
	src/cpu_traverse.cpp \
	src/gpu_traverse.cu \
	-Ibvh/src/ \
	-lassimp \
	-std=c++20 \
	-O3 \
	-x cu \
	--relocatable-device-code=true \
	-dlto \
	-arch=sm_120 \
	-Xcudafe \
		--diag_suppress=esa_on_defaulted_function_ignored \
	-o bvh.out

run:
	./bvh.out
