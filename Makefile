
include ../../work_space/grante-1.0/src/Makefile.config

OBJS=

###
all:	test

clean:
	rm -f *.o
	rm -f *.exe

###
### Production build targets
###
test: ../../work_space/grante-1.0/src/libgrante.a test.cpp
	$(CPP) $(CPPFLAGS) $(INCLUDE) -I../../work_space/grante-1.0/src -o test test.cpp \
		$(OBJS) ../../work_space/grante-1.0/src/libgrante.a $(BOOST_LIB)

