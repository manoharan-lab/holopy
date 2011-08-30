/* FILE: MieFieldExtension.h */

int nterms;
int nstop (float x);
void Dn2 (int nc, float x, float re_m, float im_m, float *re_dns, float *im_dns);
void AS( int nc, float x, float *ASpsi, float *ASeta);
void a_and_b(float x, float re_m, float im_m, int nc, float *re_a, float *im_a, 
		float *re_b, float *im_b, float *ASpsi, float *ASeta, float *re_dns, float *im_dns);
void *flds (double *re_Ec1, double *re_Ec2, double *re_Ec3,
		   double *im_Ec1, double *im_Ec2, double *im_Ec3, 
		   int nc, float xstart, int numx, float xstep,
		   float ystart, int numy, float ystep,
		   float k, float z,
		   float *re_a, float *im_a, 
		   float *re_b, float *im_b);
void *scattering_matrix (double *re_S1, double *re_S2,
		   double *im_S1, double *im_S2, 
		   int nc, float xstart, int numx, float xstep,
		   float ystart, int numy, float ystep,
		   float k, float z,
		   float *re_a, float *im_a, 
		   float *re_b, float *im_b);
void miehol(double *result, double z, float a,
			float nm,
			int xdim, int ydim, double mpp,
			float lam,
			double alpha,
			double *re_Ec1, double *im_Ec1, double xs, double ys);
void miehol_dimer(double *result, double z, double z2, float a,
				  float nm, int xdim, int ydim, double mpp,
				  float lam, double alpha, 
				  double *re_Ec1, double *im_Ec1,
				  double *re_Ec1_2, double *im_Ec1_2,
				  double *re_Ec2, double *im_Ec2,
				  double *re_Ec2_2, double *im_Ec2_2);
extern void outputhol (double xstart, double ystart, double zparam, double re_np,
		double re_nm, double radius, double alpha, int x_dim, int y_dim, double mpp, 
		float wlength, char* fn);
void *read_hol_bin(double *holdata, char *fn, int x_dim, int y_dim);
float hologramFitError (double xstart, double ystart, double zparam, double re_np,
		double re_nm, double radius, double alpha, int n, int x_dim, int y_dim, double mpp, 
		float wlength, char* input_data_fn);
