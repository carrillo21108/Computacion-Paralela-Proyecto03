# the executable name
TARGET = houghBase
SRC = houghBase.cu

all: pgm.o	$(TARGET) run

houghBase:	$(SRC) pgm.o
	nvcc $(SRC) pgm.o -o $(TARGET)

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o


# Rule to run the executable with 4 processes
run: $(TARGET)
	./$(TARGET) runway.pgm

# Clean the directory by removing the compiled executable
clean:
	rm -f $(TARGET)
