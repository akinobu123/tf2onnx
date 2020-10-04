CC = g++
CFLAGS = -std=c++11 -msse4.2 -mbmi2
LDFLAGS = -lonnxruntime
INCLUDE = -I. -I../onnxruntime-1.5.1/include -I../onnxruntime-1.5.1/include/onnxruntime/core/session
LIB = -L../onnxruntime-1.5.1/lib

target = mnist
sources = image_tensor.cxx mnist.cxx
objects = $(addprefix obj/, $(sources:.cxx=.o))

$(target): $(objects)
	$(CC) -o $@ $^ $(LIB) $(LDFLAGS) $(CFLAGS)

obj/%.o: %.cxx
	@[ -d obj ] || mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

all: $(target)

clean:
	rm -f $(objects) $(target)

