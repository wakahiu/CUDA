Include = -I/usr/include/OpenEXR
Link = -lIlmImf -lImath -lHalf
Flags = -g -G -o
Sources = *.cu *.cc
Out = prog_out
Arch = -arch=sm_11
NVCC = /usr/local/cuda/bin/nvcc
NVgdb = /usr/local/cuda/bin/cuda-gdb

all: main

# Compiling
main:
	$(NVCC) $(Sources) $(Flags) $(Out) $(Include) $(Link)  $(Arch)
	./$(Out) "stillife.exr" 9

clean:
	rm -f $(Out)
	
dbg:
	$(NVgdb) $(OUT) 
