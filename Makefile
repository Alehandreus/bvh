build_cpu:
	g++ \
	examples/example.cpp \
	src/bvh.cpp \
	src/utils.cpp \
	-lassimp \
	-fopenmp \
	-O3 \
	-o bvh

build_gpu:
	nvcc \
	examples/example.cu \
	src/bvh.cpp \
	src/utils.cpp \
	-lassimp \
	-O3 \
	-x cu \
	--relocatable-device-code=true \
	-Xcudafe \
		--diag_suppress=esa_on_defaulted_function_ignored \
	-o bvh
# -o bvh
# https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

run:
	./bvh
