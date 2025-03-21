build_cpu:
	g++ \
	examples/example_cpu.cpp \
	src/utils.cpp \
	src/build.cpp \
	src/cpu_traverse.cpp \
	-lassimp \
	-fopenmp \
	-O3 \
	-o bvh

build_gpu:
	nvcc \
	examples/example_gpu.cu \
	src/utils.cpp \
	src/build.cpp \
	src/cpu_traverse.cpp \
	src/gpu_traverse.cu \
	-lassimp \
	-O3 \
	-x cu \
	--relocatable-device-code=true \
	-dlto \
	-arch=sm_89 \
	-Xcudafe \
		--diag_suppress=esa_on_defaulted_function_ignored \
	-o bvh

run:
	./bvh
