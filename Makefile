# Makefile

NVCC := nvcc
CUDA_FLAGS := -arch=sm_75

# Directories for source files
SOURCE_DIRS := general gpuSpec

# Collect source files from the specified directories
CU_SOURCES := $(wildcard $(addsuffix /*.cu, $(SOURCE_DIRS)))
H_SOURCES := $(wildcard $(addsuffix /*.h, $(SOURCE_DIRS)))
OBJECTS := $(CU_SOURCES:.cu=.o)

# Output directory
OUT_DIR := bin

# Executable name
EXECUTABLE := pisonGPU

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	mkdir -p $(OUT_DIR)
	$(NVCC) $(CUDA_FLAGS) $(OBJECTS) -o $(OUT_DIR)/$(EXECUTABLE) testing/gpuMain.cu

%.o: %.cu $(H_SOURCES)
	$(NVCC) $(CUDA_FLAGS) -c $< -o  $@

clean:
	rm -f $(OBJECTS)
	rm -rf $(OUT_DIR)
