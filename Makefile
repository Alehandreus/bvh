debug:
	g++ src/main.cpp src/bvh.cpp  -lassimp -O3 -D DEBUG -g -o bvh
release:
	g++ src/main.cpp src/bvh.cpp  -lassimp -O3 -o bvh
run:
	./bvh
