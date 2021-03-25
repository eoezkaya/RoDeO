TARGET_EXEC := rodeo

BUILD_DIR := ./build
SRC_DIRS := ./src

# Find all the C and C++ files we want to compile
SRCS := $(shell find $(SRC_DIRS) -name *.cpp)


OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)


DEPS := $(OBJS:.o=.d)


INC_DIRS := $(shell find $(SRC_DIRS) -type d)

INC_FLAGS := $(addprefix -I,$(INC_DIRS))


CPPFLAGS := $(INC_FLAGS) -MMD -MP -fopenmp -O2
LDFLAGS :=  -lm -larmadillo -lgomp 
# The final build step.
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)


# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

# Include the .d makefiles. The - at the front suppresses the errors of missing
# Makefiles. Initially, all the .d files will be missing, and we don't want those
# errors to show up.
-include $(DEPS)