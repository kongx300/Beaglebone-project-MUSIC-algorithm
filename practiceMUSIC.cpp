#include <iostream>
#include <armadillo>
#include <math.h>   // This gives INFINITY
#include <typeinfo>
#include <stack>
#include <ctime>



#ifdef __cplusplus 
extern "C" {
#endif
    /* Declarations of this file */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <signal.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "adcdriver_host.h"
#include "spidriver_host.h"
#ifdef __cplusplus
}
#endif


using namespace std;
using namespace arma;

const float pi = 3.1415926535897932384;
#define NUM_FRAMES 100
#define NUM_PTS 10
float u[NUM_PTS];
// Stuff used with "hit return when ready..." 
char dummy[8];

stack<clock_t> tictoc_stack;


void music_estimate(float v0[NUM_PTS], float fsamp);
float music_sum(float f, fmat V);
float cross_type_product(cx_fvec cv, fvec dv);
void tic();
void toc();




int main(int argc, char** argv){
	// Buffers for tx and rx data from A/D registers
	uint32_t tx_buf[3];
	uint32_t rx_buf[4];
	
	// Initialize A/D converter
	adc_config();
	// Check the A/D is alive by reading from its config reg.
	printf("--------------------------------------------------\n");
	printf("About to read A/D config register\n");
	printf("Hit return when ready -->\n");
	fgets (dummy, 8, stdin);
	rx_buf[0] = adc_get_id_reg();
	printf("Read ID reg.  Received ID = 0x%08x\n", rx_buf[0]);
	
	// Set sample rate
	printf("--------------------------------------------------\n");
	printf("Set sample rate and set channel 0\n");
	adc_set_samplerate(SAMP_RATE_50);
	adc_set_chan0();
	
	printf("--------------------------------------------------\n");
	printf("Now read training set: %d data frames\n", NUM_FRAMES);
	printf("Hit return when ready -->\n");
	fgets (dummy, 8, stdin);
  
	// Do multiple read
	printf("--------------------------------------------------\n");
	printf("Now try to do multiple read\n");
	printf("Hit return when ready -->\n");
	fgets (dummy, 8, stdin);
	adc_read_multiple(NUM_PTS, u); 
						printf("--------------------------------------------------\n");
						printf("u[] is\n");
						for(int i = 0; i< NUM_PTS; i++){
								cout << u[i] << endl;
						}
	
	float fsamp = 31250;
	
	//create a time fvector of a signal;
	/*
    	int Nx = 1024;
    	int Tmax = 100;
    	fvec t = linspace<fvec>(0,Tmax*(Nx-1)/Nx, Nx); // t is col fvector
    	cout << "T is" << t << endl;
    	float dt;
    	dt = t(1) - t(0);
    	float fsamp = 1/dt;
    	float f0 = 2.0;
    	fvec x;
    
    	x = sin(2*pi*f0*t); // x is col fvector
	*/
	
	printf("--------------------------------------------------\n");
	printf("Preparations are done and now we are going to the main part of the algorithm\n");
	printf("Hit return when ready -->\n");
	fgets (dummy, 8, stdin);
	
	music_estimate(u,fsamp);
}


void music_estimate(float v0[NUM_PTS], float fsamp){ 
						
	int M = NUM_PTS; // M = size(v,0);
	int p = 2;
	
	// Create a vector to receive data in the float array 
	// Notice that the initialization "fvec v = zeros<fvec>(M)" is not satisfied
	fvec v(M);
	for(int i = 0; i < NUM_PTS; i++){
		v(i) = v0[i];
	}
	cout << "v is" << v << endl;				
						
	
	tic();
	fmat Rxx = v * v.t();
	toc();
	
	cout << Rxx << endl;
	
	// svd decomp
	fmat U; fmat V; fvec s;
	svd(U,s,V,Rxx);
	
	// Save the noise fvector in V
	fmat Vnoise = V.cols(p+1, size(V,1)-1);
	
	// Loop over all frequencies
	int Nf = 3001;
	fvec f = linspace<fvec>(0.0f, fsamp/2.0f, Nf);
	
	// Pmu is 3001x1.
	fmat Pmu = zeros<fvec>(Nf, 1); // size(.) returns a 2d size but .size() return 1d
	
	for(int i = 0; i < Nf; i++){
		Pmu(i) = music_sum(f(i)/fsamp, Vnoise);
	}
	
	uvec idxM = zeros<uvec>(Nf);
	fvec Pcol;
	for(int i=0; i<size(Pmu,1); i++){
		
		Pcol = Pmu.col(i);
		// Find index of max value of Pcol
		float elt = -INFINITY;
		int maxj = 0;
		for (int j=0; j<Nf; j++) {
	    	if (Pcol(j) > elt) {
	    	maxj = j;
	    	elt = Pcol(j);
	    	}
		}
		idxM(i) = maxj;
	}
	
	fvec f0(idxM.size());
	for(int i = 0; i< f0.size(); i++){
		f0(i) = f(idxM(i));
	}
	
	cout << scientific;
	cout << "Estimated frequency is " << f0(0) << endl;
	
	
}

float music_sum(float f, fmat V){
	
	int Mr = size(V,0);
	cx_fvec e(Mr);
	cx_fvec e0(Mr);
	
	float idx = 0;
	for(int i = 0; i<Mr; i++){
		e0(i) = cx_float(0.0, 2*pi*idx*f);
		idx += 1;
	}
	e = exp(e0);
	
	int Mc = size(V,1); 
	
	float s = 0;
	float ele = 0;
	for(int i = 0; i<Mc; i++){
		//t = e.as_row() * V.col(i);
		//s += abs(t)*abs(t);
		// Instead of jumping to the complex math， 
		// we define a new function to do the complex product
		ele = cross_type_product(e,V.col(i));
		s += ele * ele;
	}
	
	float p = 1/s;
	return p;	
}


float cross_type_product(cx_fvec cv, fvec dv){
	// We want to get cv * dv
	// We first split the complex fvector to two real fvecotors(real & imag parts)
	// Then do the product in real math
	
	
	float re;
	float im;
	frowvec cv_re(cv.size()); // real part of the complex fvector
	frowvec cv_im(cv.size()); // imaginary part of the complex fvector
	for(int i =0; i<cv.size(); i++){
		re = cv(i).real(); // extract the real part
		im = cv(i).imag(); // extract the imaginary part
		cv_re(i) = re;
		cv_im(i) = im;	
	}
	cx_float s = cx_float(0,0); // Define a new complex number to 
								  // save the results of the multip
	for(int i =0; i<cv.size(); i++){
		s += cx_float(cv_re(i)*dv(i),cv_im(i)*dv(i));
	}
	// Norm of the complex number 
	float rel = norm(s);
	
	return rel;
}

void tic() {
    tictoc_stack.push(clock());
}

void toc() {
    cout << "Time elapsed: "
              << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
              << endl;
    tictoc_stack.pop();
}