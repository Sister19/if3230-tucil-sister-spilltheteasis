EXECUTE_FOLDER	= bin
ALGORITHM		= dft
NUM_PROCESSES	= 4
INPUT_FILE		= 128.txt
OUTPUT_FILE		= output.txt

all: serial parallel

open-mpi:
	mpicc src/open-mpi/open-mpi-$(ALGORITHM).c -o $(EXECUTE_FOLDER)/open-mpi-$(ALGORITHM) -lm && \
	time mpirun -np $(NUM_PROCESSES) $(EXECUTE_FOLDER)/open-mpi-$(ALGORITHM) < test_case/$(INPUT_FILE) > out/$(OUTPUT_FILE)

open-mp:
	gcc src/open-mp/open-mp-$(ALGORITHM).c -o $(EXECUTE_FOLDER)/open-mp-$(ALGORITHM) -lm -fopenmp && \
	time $(EXECUTE_FOLDER)/open-mp-$(ALGORITHM) < test_case/$(INPUT_FILE) > out/$(OUTPUT_FILE)

serial:
	gcc src/serial/c/serial-$(ALGORITHM).c -o $(OUTPUT_FOLDER)/serial-$(ALGORITHM) -lm