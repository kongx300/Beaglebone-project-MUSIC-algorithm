#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;


int main(int argc, char** argv){
	double pi = 3.1415926535897932384;
	
	
	//create a time vector of a signal;
	int Nx = 136;
	int Tmax = 13;
	vec t = linspace(0,Tmax*(Nx-1)/Nx, Nx);
	
	double dt;
	dt = t(1) - t(0);
	double fsamp = 1/dt;
	double f0 = 1.0;
	vec x;
	x = sin(2*pi*f0*t);
	
	cout << "x is" << x << endl;
	
	music_estimate(x,fsamp);
}


void music_estimate(vec v, double fsamp){
	int M;
	M = v.size(); // M = size(v,0);
	
	int p = 2;
	
	// Covariance matrix
	mat Rxx;
	Rxx = v * v.as_row();
	// svd decomp
	mat U; mat V; vec s;
	svd(U,s,V,Rxx);
	
	// Save the noise vector in V
	mat Vnoise = V.cols(p+1, V.size()-1);
	
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
	
	cout << "Estimated frequency is " << f0(0) << endl;
}


double music_sum(double f, mat V){
	double pi = 3.1415926535897932384;
	
	int Mr = size(V,0);
	cx_vec e(Mr), e0(Mr);
	
	double idx = 0;
	for(int i = 0; i<Mr; i++){
		e0(i) = cx_double(0.0, 2*pi*idx*f);
		idx += 1;
	}
	e = exp(e0);
	
	int Mc = size(V,1);
	int s = 0;
	double t;
	for(int i = 0; i<Mc; i++){
		t = e.as_row() * V.cols(0,i);
		s += abs(t)*abs(t);
	}
	
	double p = 1/s;
	return p;	
}
