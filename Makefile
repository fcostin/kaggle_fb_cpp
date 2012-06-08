CPPFLAGS := -Wall -pedantic -O3


test:	a.out
	./a.out
.IGNORE: test

clean:
	rm -rf ./a.out
	rm -rf *.o
.IGNORE: clean

a.out:	main.cpp
	g++ --std=c++11 -lm $(CPPFLAGS) $^ -o $@
