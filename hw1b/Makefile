Include = -I/usr/include/OpenEXR
Link = -lIlmImf -lImath -lHalf
Flags = -I. -o
Sources = *.cu *.cc
Out = prog_out
Arch = -arch=sm_11
NVCC = /usr/local/cuda/bin/nvcc

all: main

# Compiling
main:
	$(NVCC) $(Sources) $(Flags) $(Out) $(Include) $(Link)  $(Arch)
	./$(Out) "stillife.exr" 5

clean:
	rm -f $(Out)
