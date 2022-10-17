SRCDIR := source
INCDIR := include
BUILDDIR := build
TARGET := bin/main

# Other variables
SOURCES := $(shell find $(SRCDIR) -type f -name "*.cpp")
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.cpp=.o))
SOURCESCU := $(shell find $(SRCDIR) -type f -name "*.cu")
OBJECTSCU := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCESCU:.cu=.cu.o))
TARGETDIR = `dirname $(TARGET)`

# Compilers
HOST_COMPILER := g++
NVCC          := nvcc #-ccbin $(HOST_COMPILER)  #/usr/local/cuda/bin/nvcc

# LIBS path



# Flags
NVCCFLAGS     := -m64 -std=c++11 -O3 -cudart=shared#-dc used cg::grid_group, cg::this_group
CCFLAGS       := -std=c++11 
LDFLAGS       := 
LIKWID_DEFINES := -DLIKWID_NVMON
# Debug build flags
ifeq ($(dbg),1)
    NVCCFLAGS += -g -G
    BUILD_TYPE := debug
else
    BUILD_TYPE := release
endif

# Main flags
ALL_CCFLAGS := $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_LDFLAGS := $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Includes and libraries
CUDA_PATH := /local/cuda-10.1/toolkit
LIBPATH := $(CUDA_PATH)/lib64
SOLIBS := $(LIBPATH)/libcublas.so

INCLUDES      := $(addprefix -I ,$(shell find $(SRCDIR) -type d))
LIBRARIES     := -lcublas

# LIKWID_LIB := -L$(PREFIX)/lib
# LIKWID_INC := -I$(PREFIX)/include

ifneq ($(INCDIR),)
	INCLUDES += -I $(INCDIR)
endif

################################################################################

# Gencode arguments
SMS ?= 70 72 75

# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif

################################################################################

# Target rules
all: build

build: $(TARGET)

run: build
	./$(TARGET)

clean:
	rm -fr $(OBJECTS) $(OBJECTSCU) $(TARGET)

clobber:
	rm -fr $(BUILDDIR) $(TARGETDIR)

$(BUILDDIR)/%.cu.o: $(SRCDIR)/%.cu
	@mkdir -p $(BUILDDIR);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(TARGET): $(OBJECTS) $(OBJECTSCU)
	@mkdir -p $(TARGETDIR);
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(SOLIBS) $(LIBRARIES)
