CC := gcc
CPP := g++
CFLAGS := -O3 -mfpu=vfpv3 -mfloat-abi=hard -march=armv7 -v -I./include 
INCLUDEDIR := -I./include
INCLUDES := prussdrv.h pru_types.h __prussdrv.h pruss_intc_mapping.h spidriver_host.h adcdriver_host.h

LIBLOCS := -Wl,-rpath,/usr/local/lib -L/usr/local/lib
LDFLAGS := -lm -lstdc++ -larmadillo -lblas -lcblas -llapack \
 -lgfortran


SRCS := practiceMUSIC.cpp prussdrv.c adcdriver_host.c spidriver_host.c
OBJS := practiceMUSIC.o prussdrv.o adcdriver_host.o spidriver_host.o
EXES := practiceMUSIC

#----------------------------------------------------
# PRU code
DEVICE=am335x
PRU_CC := clpru
PRU_CC_FLAGS := --silicon_version=3 -I./include -I/usr/share/ti/cgt-pru/include/ -D$(DEVICE) -i/usr/share/ti/cgt-pru/lib
PRU_LINKER_SCRIPT := AM335x_PRU.cmd
PRU_INCLUDES := resource_table_empty.h pru_ctrl.h pru_intc.h pru_cfg.h pru_spi.h

# PRU0 is the SPI stuff.
PRU0_SRCS := pru0.c pru_spi.c
PRU0_OBJS := pru0.obj pru_spi.obj
PRU0_MAP := pru0.map
PRU0_EXES := data0.bin text0.bin

# PRU1 is the oscillator
PRU1_SRCS := pru1.c
PRU1_OBJS := pru1.obj
PRU1_MAP := pru1.map
PRU1_EXES := data1.bin text1.bin

PRU_HEXPRU_SCRIPT := bin.cmd

#=================================================
all: practiceMUSIC pru0.bin pru1.bin 

# Compile practiceMUSIC.cpp
practiceMUSIC.o: practiceMUSIC.cpp 
	echo "--> Building practiceMUSIC.o"
	$(CPP) $(CFLAGS) $(INCLUDEDIR) -c $< -o $@

# Link 
practiceMUSIC: $(OBJS) 
	echo "--> Linking ...."
	$(CPP) -std=c++0x -Wall -g3 $(CFLAGS) $(LIBLOCS) $(LDFLAGS) $^ -o $@
	#g++ -std=c++0x -Wall -O0 -g3 psinkt.cpp -o ./psinkt.out
#$(OBJS): $(INCLUDES)


# ---------------------
adcdriver_host.o: adcdriver_host.c ./include/adcdriver_host.h
	echo "--> Building adcdriver_host.o"
	$(CC) $(CFLAGS) $(INCLUDEDIR) -c $< -o $@

spidriver_host.o: spidriver_host.c ./include/spidriver_host.h
	echo "--> Building spidriver_host.o"
	$(CC) $(CFLAGS) $(INCLUDEDIR) -c $< -o $@

prussdrv.o: prussdrv.c # $(DEPS)
	echo "--> Building prussdrv.o"
	$(CC) $(CFLAGS) $(INCLUDEDIR) -c $< -o $@


#--------------------------------
# Compile and link the PRU sources to create ELF executable
pru0.out: pru0.c pru_spi.c
	echo "--> Building and linking PRU0 stuff..."
	$(PRU_CC) $^ $(PRU_CC_FLAGS) -z $(PRU_LINKER_SCRIPT) -o $@ -m $(PRU0_MAP)

# Build PRU .bin file from ELF
pru0.bin: pru0.out $(PRU_HEXPRU_SCRIPT)
	echo "--> Running hexpru for PRU0..."
	hexpru $(PRU_HEXPRU_SCRIPT) $<
	-mv text.bin text0.bin
	-mv data.bin data0.bin

pru1.out: pru1.c
	echo "--> Building and linking PRU1 stuff..."
	$(PRU_CC) $^ $(PRU_CC_FLAGS) -z $(PRU_LINKER_SCRIPT) -o $@ -m $(PRU1_MAP)

# Build PRU .bin file from ELF
pru1.bin: pru1.out $(PRU_HEXPRU_SCRIPT)
	echo "--> Running hexpru for PRU1..."
	hexpru $(PRU_HEXPRU_SCRIPT) $<
	-mv text.bin text1.bin
	-mv data.bin data1.bin

#--------------------------------
# Clean
clean:
	-rm -f *.o *.obj *.out *.map $(EXES) $(OBJS) *~ 


