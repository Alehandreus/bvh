build_gpu:
	nvcc \
	examples/example.cu \
	src/bvh.cu \
	src/utils.cu \
	-lassimp \
	-O3 \
	-x cu \
	--relocatable-device-code=true \
	-Xcudafe \
		--diag_suppress=esa_on_defaulted_function_ignored \
	-o bvh

run:
	./bvh
