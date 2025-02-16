debug:
	g++ src/main.cpp src/bvh.cpp src/utils.cpp -lassimp -O3 -D DEBUG -g -o bvh
release:
	g++ src/main.cpp src/bvh.cpp src/utils.cpp -lassimp -O3 -Wno-narrowing -o bvh
run:
	./bvh
