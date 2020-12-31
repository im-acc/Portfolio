# include <math.h>

const float PI = 3.14159265f;

// non-homogenous terme
float f(float x, float y) {
	return sin(x*PI)*sin(y*PI);
}

// Solver laplace(v) = f(x,y)
// [0,1]x[0,1] Square boundary valued at 0 ( v(B) = 0 ) 
void host_solver(float *v, int n, int niter) {
	
	float h = 1 / ((float)n); // step size
	float h_2 = h * h;
	int d = n + 2; // dim length
	// initialize grid with 0 (f(x,y) also would be possible)
	int i_max = (n + 2) * (n + 2);
	for (int i = 0; i < i_max; i+=d)
	{
		for (int j = 0; j < n + 2; j++)
		{
			v[i + j] = 0.0f;
		}
	}
	// Jacobi iterations
	i_max = (n + 1) * (n + 1);
	float x=h;
	float y=h;
	for (int k = 0; k < niter; k++)
	{
		for (int i= d;  i < i_max; i+=d)
		{
			for (int j = 1; j < n+1; j++)
			{
				v[i+j] = (v[i-d+j] + v[i+j-1] + v[i+d+j] + v[i+j+1] - h_2 * f(x,y))*0.25;
				y += h;
			}
			x += h;
		}
	}
}