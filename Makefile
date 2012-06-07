test:	a.out
	./a.out
.IGNORE: test

clean:
	rm -rf ./a.out
	rm -rf *.o
.IGNORE: clean

a.out:	main.cpp
	g++ --std=c++11 -lm -O3 $^ -o $@
