#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

const double pi = 3.1415926535897932384;


void music_estimate(vec v, double fsamp);
double music_sum(double f, mat V);
double cross_type_product(cx_vec cv, vec dv);


int main(int argc, char** argv){
	
	//create a time vector of a signal;
	int Nx = 136;
	int Tmax = 13;
	vec t = linspace(0,Tmax*(Nx-1)/Nx, Nx); // t is col vector
	
	double dt;
	dt = t(1) - t(0);
	double fsamp = 1/dt;
	double f0 = 2.0;
	vec x;
	x = sin(2*pi*f0*t); // x is col vector
	
	//cout << "x is" << x << endl;
	
	music_estimate(x,fsamp);
}


void music_estimate(vec v, double fsamp){
	int M;
	M = v.size(); // M = size(v,0);
	
	int p = 2;
	
	// Covariance matrix
	mat Rxx;
	Rxx = v * v.t();
	// svd decomp
	mat U; mat V; vec s;
	svd(U,s,V,Rxx);
	
	// Save the noise vector in V
	mat Vnoise = V.cols(p+1, size(V,1)-1);
	
	// Loop over all frequencies
	int Nf = 3001;
	vec f = linspace(0,fsamp/2, Nf);
	mat Pmu = zeros(size(f)); // size(.) returns a 2d size but .size() return 1d
	
	for(int i = 0; i < Nf; i++){
		Pmu(i) = music_sum(f(i)/fsamp, Vnoise);
	}
	// here comes the figure 2;
	
	int j;
	vec idxM(size(Pmu,1));
	for(int i=0; i<size(Pmu,1); i++){
		j = (Pmu.col(i)).index_max();
		idxM(i) = j;
	}
	
	vec f0(idxM.size());
	for(int i = 0; i< f0.size(); i++){
		f0(i) = f(idxM(i));
	}
	
	cout << scientific;
	cout << "Estimated frequency is " << f0(0) << endl;
}

double music_sum(double f, mat V){
	
	int Mr = size(V,0);
	cx_vec e(Mr);
	cx_vec e0(Mr);
	
	double idx = 0;
	for(int i = 0; i<Mr; i++){
		e0(i) = cx_double(0.0, 2*pi*idx*f);
		idx += 1;
	}
	e = exp(e0);
	
	int Mc = size(V,1);
	double s = 0;
	double ele = 0;
	for(int i = 0; i<Mc; i++){
		//t = e.as_row() * V.col(i);
		//s += abs(t)*abs(t);
		// Instead of jumping to the complex math， 
		// we define a new function to do the complex product
		ele = cross_type_product(e,V.col(i));
		s += ele * ele;
	}
	
	double p = 1/s;
	return p;	
}

double cross_type_product(cx_vec cv, vec dv){
	// We want to get cv * dv
	// We first split the complex vector to two real vecotors(real & imag parts)
	// Then do the product in real math
	
	
	double re;
	double im;
	rowvec cv_re(cv.size()); // real part of the complex vector
	rowvec cv_im(cv.size()); // imaginary part of the complex vector
	for(int i =0; i<cv.size(); i++){
		re = cv(i).real(); // extract the real part
		im = cv(i).imag(); // extract the imaginary part
		cv_re(i) = re;
		cv_im(i) = im;	
	}
	cx_double s = cx_double(0,0); // Define a new complex number to 
								  // save the results of the multip
	for(int i =0; i<cv.size(); i++){
		s += cx_double(cv_re(i)*dv(i),cv_im(i)*dv(i));
	}
	// Norm of the complex number 
	double rel = norm(s);
	
	return rel;
}
