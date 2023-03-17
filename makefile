EXECUTE_FOLDER	= bin
ALGORITHM		= dft
NUM_PROCESSES	= 4
INPUT_FILE		= 128.txt
OUTPUT_FILE		= output

all: serial parallel

open-mpi:
	mpicc src/open-mpi/open-mpi-$(ALGORITHM).c -o $(EXECUTE_FOLDER)/open-mpi-$(ALGORITHM) -lm && \
	time mpirun -np $(NUM_PROCESSES) $(EXECUTE_FOLDER)/open-mpi-$(ALGORITHM) < test_case/$(INPUT_FILE) > out/$(OUTPUT_FILE)-open-mpi-$(ALGORITHM).txt

open-mp:
	gcc src/open-mp/open-mp-$(ALGORITHM).c -o $(EXECUTE_FOLDER)/open-mp-$(ALGORITHM) -lm -fopenmp && \
	time $(EXECUTE_FOLDER)/open-mp-$(ALGORITHM) < test_case/$(INPUT_FILE) > out/$(OUTPUT_FILE)-open-mp-$(ALGORITHM).txt

cuda:
	nvcc src/cuda/cuda-$(ALGORITHM).cu -o $(EXECUTE_FOLDER)/cuda-$(ALGORITHM) && \
	time $(EXECUTE_FOLDER)/cuda-$(ALGORITHM) < test_case/$(INPUT_FILE) > out/$(OUTPUT_FILE)-cuda-$(ALGORITHM).txt

serial:
	gcc src/serial/c/serial-$(ALGORITHM).c -o $(OUTPUT_FOLDER)/serial-$(ALGORITHM) -lm && \
	time $(OUTPUT_FOLDER)/serial-$(ALGORITHM) < test_case/$(INPUT_FILE) > out/$(OUTPUT_FILE)-serial-$(ALGORITHM).txt