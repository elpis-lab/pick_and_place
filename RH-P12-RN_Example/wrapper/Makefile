CXX = g++
CXXFLAGS = -std=c++11 -fPIC -I../win64/DynamixelSDK_x64Lib/include
LDFLAGS = -shared -L../linux64 -ldxl_x64_cpp

TARGET = librh_p12_rna.so
SRCS = rh_p12_rna_wrapper.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)