# g++
WARNING_FLAGS := -Wall -Wextra -Werror -pedantic -Wconversion
OPT_FLAGS := -O2 -g
CPPC_FLAGS := $(WARNING_FLAGS) $(OPT_FLAGS) -mtune=core2
# for google-perftools profiler
# ref : http://stackoverflow.com/questions/9577627/ubuntu-11-10-linking-perftools-library
WITH_GPERFTOOLS_PROFILING := -Wl,--no-as-needed -lprofiler -Wl,-as-needed
WITH_GPERFTOOLS_MALLOC := -Wl,--no-as-needed -ltcmalloc -Wl,-as-needed
TOOLING := $(WITH_GPERFTOOLS_PROFILING) $(WITH_GPERFTOOLS_MALLOC)
CPPC := g++

all: a.out
.PHONY: all

a.out: main.o
	$(CPPC) $(CPPC_FLAGS) $(TOOLING) -o $@ $^

%.o:	%.cpp
	$(CPPC) $(CPPC_FLAGS) -c $^

clean:
	rm -rf *.out
	rm -rf *.o
.PHONY: clean
