debug:
	g++ examples/example.cpp src/bvh.cpp src/utils.cpp -lassimp -fopenmp -O3 -D DEBUG -g -o bvh
release:
	g++ examples/example.cpp src/bvh.cpp src/utils.cpp -lassimp -fopenmp -O3 -Wno-narrowing -o bvh
run:
	./bvh
