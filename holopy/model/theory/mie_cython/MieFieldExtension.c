/* FILE: MieFieldExtension.c */

/****************************************
For extending the Mie calculations from C
into python modules.

AUTHORS: RYAN MCGORTY 

CREATED ON 12/30/2009
***************************************/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "MieFieldExtension.h"

int nterms;
float imm_glob;
int sa_correct = 0;
int xs_sa = 0;
int ys_sa = 0;
int planecorrect = 0;
int sa_param;
int sa_coeff;
//double glob_mpp = 0.0681818;
int to_weight = 0;


//Determine how many terms to include
int nstop ( float x)
{
	//x is the size parameter
	float nc;

	nc = x + 8 + (4.05 * (pow(x,1/3)));
	
	return (int)ceil(nc);
}


void Dn2 ( int nc, float x, float re_m, float im_m, float *re_dns, float *im_dns)
{
	double re_z, im_z;
	double re_oneoverz, im_oneoverz;
	double tmp1, tmp2;
	double real_den, imag_den;
	int ii = 0;

	re_z = x * re_m;
	im_z = x * im_m;

	tmp1 = pow(re_z,2)+pow(im_z,2);
	re_oneoverz = re_z / tmp1;
	im_oneoverz = (-1.) * im_z / tmp1;
	
	for (ii=0; ii<nc; ii++)
	{
		re_dns[ii]=0;
		im_dns[ii]=0;
	}
	for (ii=nc-2; ii>=0; ii--)
	{
		tmp1 = re_oneoverz * (1.+ii);
		real_den = re_dns[ii+1] + tmp1;
		imag_den = im_dns[ii+1] - ((1.+ii)*im_oneoverz);
		tmp2 = pow(real_den,2) + pow(imag_den,2);
		re_dns[ii] = tmp1 - (real_den / tmp2);
		im_dns[ii] = (-1.*im_oneoverz*(1.+ii)) + (imag_den / tmp2);
	}
}

void AS( int nc, float x, float *ASpsi, float *ASeta)
{
	int ii, n;
	int term=0;
	float u;

	double an;
	double fnn;
	double fdn;
	double pnn;
	double pdn;

	double xin;
	double xid;
	double amp2;
	double zd;
	double zn;

	long double tmp1, tmp2;

	int jumpnn=0;
	int jumpdn=0;

	ASpsi[0] = (float)sin(x);
	ASeta[0] = (float)(-1.) * cos(x);

	for (ii = 1; ii < nc; ii++)
	{
		u = 0.5 + ii;
		term = 0;

		an = 2. * u / x;

		fnn = an;
		fdn = 1.;
		pnn = an;
		pdn = (double)pow((double)10.,(double)80.); //just a big number

		jumpnn = 0;
		jumpdn = 0;

		n=2;
		
		while (term==0)
		{
			an = pow(-1.,1.+n) * 2. * (u+n-1.) / x;

			if (fabs(pnn) < 1e-6)
			{
				xin = pnn * an + 1.;
				amp2 = pow(-1.,n) * 2. * (u+n) / x;
				zn = (amp2 * xin + pnn) / xin;
				jumpnn = 3;
				pnn = 1;
			}
			if (fabs(pdn) < 1e-6)
			{
				xid = pdn * an + 1.;
				amp2 = pow(-1.,n) * 2. * (u+n) / x;
				zd = (amp2 * xid + pdn) / xid;
				jumpdn = 2;
				pdn = 1;
			}
			if (jumpnn == 0)
			{
				pnn = an + (1./pnn);
				fnn *= pnn;
			}
			if (jumpnn == 1)
			{
				pnn = zn;
				fnn *= xin;
			}
			if (jumpdn == 0)
			{
				pdn = an + (1./pdn);
				fdn *= pdn;
			}
			if (jumpdn == 1)
			{
				pdn = zd;
				fdn *= xid;
			}

			term = (fabs(pnn/pdn - 1) < 1e-18) && ((jumpdn+jumpnn)==0);

			jumpnn = jumpnn - (jumpnn != 0);
			jumpdn = jumpdn - (jumpdn != 0);

			n = n+1;
		}

		ASpsi[ii] = ASpsi[ii-1] / fnn * fdn;
		ASeta[ii] = ASeta[ii-1] * fdn / fnn - 1./ASpsi[ii-1];
		
		}

	}

void a_and_b(float x, float re_m, float im_m, int nc, 
			 float *re_a, float *im_a, 
			 float *re_b, float *im_b,
			 float *ASpsi,
			 float *ASeta, 
			 float *re_dns, 
			 float *im_dns)
{
	int i=0;
	double re_fac, im_fac;
	double tmp;
	double re_num, im_num;
	double re_den, im_den;

	re_a[0] = 0.;
	re_b[0] = 0.;
	im_a[0] = 0.;
	im_b[0] = 0.;

	for (i=1; i<nc; i++)
	{
		tmp = pow(re_m,2) + pow(im_m,2);
		re_fac = (re_dns[i] * re_m + im_dns[i] * im_m) / tmp;
		re_fac += i/x;
		im_fac = (im_dns[i] * re_m - re_dns[i] * im_m) / tmp;
		re_num = (re_fac * ASpsi[i]) - ASpsi[i-1];
		im_num = (im_fac * ASpsi[i]);
		re_den = (re_fac * ASpsi[i] - im_fac * ASeta[i]) - ASpsi[i-1];
		im_den = (re_fac * ASeta[i] + im_fac * ASpsi[i]) - ASeta[i-1];
		tmp = pow(re_den,2) + pow(im_den,2);
		re_a[i] = (re_num*re_den + im_num*im_den) / tmp;
		im_a[i] = (im_num*re_den - re_num*im_den) / tmp;

		re_fac = (re_m*re_dns[i] - im_m*im_dns[i]);
		re_fac += i/x;
		im_fac = (re_m*im_dns[i] + im_m*re_dns[i]);
		re_num = (re_fac * ASpsi[i]) - ASpsi[i-1];
		im_num = (im_fac * ASpsi[i]);
		re_den = (re_fac * ASpsi[i] - im_fac * ASeta[i]) - ASpsi[i-1];
		im_den = (re_fac * ASeta[i] + im_fac * ASpsi[i]) - ASeta[i-1];
		tmp = pow(re_den,2) + pow(im_den,2);
		re_b[i] = (re_num*re_den + im_num*im_den) / tmp;
		im_b[i] = (im_num*re_den - re_num*im_den) / tmp;
	}
}

void *flds (double *re_Ec1, double *re_Ec2, double *re_Ec3,
		   double *im_Ec1, double *im_Ec2, double *im_Ec3, 
		   int nc, float xstart, int numx, float xstep,
		   float ystart, int numy, float ystep,
		   float k, float z,
		   float *re_a, float *im_a, 
		   float *re_b, float *im_b)
{
	float *xs;
	float *ys;
	float rho, rho2;
	float *kr;
	double temp, temp1, temp2;
	double re_tmp, im_tmp;
	float *costheta;
	float *costhetanf;
	float *sintheta;
	float *cosphi;
	float *sinphi;
	double *pi_nm1;
	double *pi_n;
	double *re_xi_nm2, *im_xi_nm2;
	double *re_xi_nm1, *im_xi_nm1;
	double *re_xi_n, *im_xi_n;
	double re_Mo1n1, re_Mo1n2, re_Mo1n3;
	double im_Mo1n1, im_Mo1n2, im_Mo1n3;
	double re_Ne1n1, re_Ne1n2, re_Ne1n3;
	double im_Ne1n1, im_Ne1n2, im_Ne1n3;
	double *re_Es1, *re_Es2, *re_Es3;
	double *im_Es1, *im_Es2, *im_Es3;
	double *swisc, *twisc;
	double tau_n;
	unsigned int m=0;
	double re_dn;
	double im_dn;
	double re_En = 0;
	double im_En = 0;
	unsigned int powtemp = 1;
	unsigned int sizem = numx*numy;
	unsigned int i,j,n;

	xs = (float*)malloc(numx*sizeof(float));
	ys = (float*)malloc(numy*sizeof(float));
	kr = (float*)malloc(sizem*sizeof(float));
	costheta = (float*)malloc(sizem*sizeof(float));
	costhetanf = (float*)malloc(sizem*sizeof(float));
	sintheta = (float*)malloc(sizem*sizeof(float));
	cosphi = (float*)malloc(sizem*sizeof(float));
	sinphi = (float*)malloc(sizem*sizeof(float));
	pi_nm1 = (double*)malloc(sizem*sizeof(double));
	pi_n = (double*)malloc(sizem*sizeof(double));
	re_xi_nm2 = (double*)malloc(sizem*sizeof(double));
	re_xi_nm1 = (double*)malloc(sizem*sizeof(double));
	im_xi_nm2 = (double*)malloc(sizem*sizeof(double));
	im_xi_nm1 = (double*)malloc(sizem*sizeof(double));
	re_xi_n = (double*)malloc(sizem*sizeof(double));
	im_xi_n = (double*)malloc(sizem*sizeof(double));
	re_Es1 = (double*)malloc(sizem*sizeof(double));
	re_Es2 = (double*)malloc(sizem*sizeof(double));
	re_Es3 = (double*)malloc(sizem*sizeof(double));
	im_Es1 = (double*)malloc(sizem*sizeof(double));
	im_Es2 = (double*)malloc(sizem*sizeof(double));
	im_Es3 = (double*)malloc(sizem*sizeof(double));
	swisc = (double*)malloc(sizem*sizeof(double));
	twisc = (double*)malloc(sizem*sizeof(double));

	for (i=0; i<numx; i++)
	{
		xs[i] = xstart + (i*xstep);

		//First, I tried this.  It is correct if the index mismatch is at the
		//focal plane.  That's not the case, but even so, it did seem to help a bit.
		//xs[i] = (xs[i] * z * sa_coeff) / sqrt(pow(z,2) + (pow(xs[i],2)*(1-pow(sa_coeff,2))));


		//temp = (pow(z,2)*pow(sa_coeff,2)) + ((0.5)*zoffset*(1+pow(xs[i]/z,2))*(pow(sa_coeff,2)-1));
		//temp = temp / ((pow(xs[i],2)*(pow(sa_coeff,2)-1)) + pow(z*sa_coeff,2));
		//xs[i] = xs[i]*sqrt(temp);


		for (j=0; j<numy; j++)
		{
			ys[j] = ystart + (j*ystep);

			//ys[j] = (ys[j] * z * sa_coeff) / sqrt(pow(z,2) + (pow(ys[j],2)*(1-pow(sa_coeff,2))));

			//temp = (pow(z,2)*pow(sa_coeff,2)) + ((0.5)*zoffset*(1+pow(ys[j]/z,2))*(pow(sa_coeff,2)-1));
			//temp = temp / ((pow(ys[j],2)*(pow(sa_coeff,2)-1)) + pow(z*sa_coeff,2));
			//ys[j] = ys[j]*sqrt(temp);

			rho = sqrt(pow(xs[i],2)+pow(ys[j],2));
			rho2 = sqrt(pow(xs_sa + (i*xstep),2) + pow(ys_sa + (j*ystep),2));
			kr[m] = k*sqrt(pow(rho,2)+pow(z,2));

			if (sa_correct)
			{
				if ((ys_sa==0) && (xs_sa==0))
					kr[m] = k*sqrt(pow(rho,2)+pow(z-(sa_param*pow(rho,4)),2));
				else
					kr[m] = k*sqrt(pow(rho,2)+pow(z-(sa_param*pow(rho2,4)),2));
			}
			//Let temp be theta:
			temp = atan2(rho,z);

			if (sa_correct)
			{
				if ((ys_sa==0) && (xs_sa==0))
					temp = atan2(rho, z-(sa_param*pow(rho,4)));
				else
					temp = atan2(rho, z-(sa_param*pow(rho2,4)));
			}
			costheta[m] = cos(temp);
			costhetanf[m] = cos(temp*sa_coeff);
			sintheta[m] = sin(temp);
			//Now temp is phi:
			temp = atan2(xs[i],ys[j]);
			cosphi[m] = cos(temp);
			sinphi[m] = sin(temp);

			re_xi_nm2[m] = cos(kr[m]);
			im_xi_nm2[m] = sin(kr[m]);
			re_xi_nm1[m] = sin(kr[m]);
			im_xi_nm1[m] = (-1.)*cos(kr[m]);

			pi_nm1[m] = 0.;
			pi_n[m] = 1.;

			re_Es1[m] = 0.;
			re_Es2[m] = 0.;
			re_Es3[m] = 0.;
			im_Es1[m] = 0.;
			im_Es2[m] = 0.;
			im_Es3[m] = 0.;

			m++;
		}
	}

	for (n = 1; n<nc; n++)
	{
		temp = (n + n + 1.) / n / (1.+n);
		for (m=sizem; m--; )
		{
			
			swisc[m] = pi_n[m] * costheta[m];
			twisc[m] = swisc[m] - pi_nm1[m];
			tau_n = (n*twisc[m]) - pi_nm1[m];

			temp2 = (n + n - 1.)/kr[m];
			re_xi_n[m] = temp2*re_xi_nm1[m] - re_xi_nm2[m];
			im_xi_n[m] = temp2*im_xi_nm1[m] - im_xi_nm2[m];

			re_Mo1n1 = 0.;
			im_Mo1n1 = 0.;
			re_Mo1n2 = pi_n[m] * re_xi_n[m];
			im_Mo1n2 = pi_n[m] * im_xi_n[m];
			re_Mo1n3 = (-1.*tau_n) * re_xi_n[m];
			im_Mo1n3 = (-1.*tau_n) * im_xi_n[m];

			temp2 = n/kr[m];
			re_dn = temp2*re_xi_n[m] - re_xi_nm1[m];
			im_dn = temp2*im_xi_n[m] - im_xi_nm1[m];
			re_Ne1n1 = n*(n+1.)*pi_n[m]*re_xi_n[m];
			im_Ne1n1 = n*(n+1.)*pi_n[m]*im_xi_n[m];
			re_Ne1n2 = -1.*tau_n*re_dn;
			im_Ne1n2 = -1.*tau_n*im_dn;
			re_Ne1n3 = pi_n[m]*re_dn;
			im_Ne1n3 = pi_n[m]*im_dn;

			if (powtemp==1)
			{
				re_En = 0.;
				im_En = temp;
			}
			else if (powtemp==2)
			{
				re_En = -1. * temp;
				im_En = 0.;
			}
			else if (powtemp==3)
			{
				re_En = 0.;
				im_En = -1. * temp;
			}
			else if (powtemp==4)
			{
				re_En = temp;
				im_En = 0.;
			}

			temp1 = re_a[n] * re_Ne1n1 - im_a[n] * im_Ne1n1;
			temp2 = re_a[n] * im_Ne1n1 + im_a[n] * re_Ne1n1;
			re_tmp = (-1.*temp2) - (re_b[n]*re_Mo1n1 - im_b[n]*im_Mo1n1);
			im_tmp = temp1 - (re_b[n]*im_Mo1n1 + im_b[n]*re_Mo1n1);
			re_Es1[m] += (re_En * re_tmp) - (im_En * im_tmp);
			im_Es1[m] += (im_En * re_tmp) + (re_En * im_tmp);

			temp1 = re_a[n] * re_Ne1n2 - im_a[n] * im_Ne1n2;
			temp2 = re_a[n] * im_Ne1n2 + im_a[n] * re_Ne1n2;
			re_tmp = (-1.*temp2) - (re_b[n]*re_Mo1n2 - im_b[n]*im_Mo1n2);
			im_tmp = temp1 - (re_b[n]*im_Mo1n2 + im_b[n]*re_Mo1n2);
			re_Es2[m] += (re_En * re_tmp) - (im_En * im_tmp);
			im_Es2[m] += (im_En * re_tmp) + (re_En * im_tmp);

			temp1 = re_a[n] * re_Ne1n3 - im_a[n] * im_Ne1n3;
			temp2 = re_a[n] * im_Ne1n3 + im_a[n] * re_Ne1n3;
			re_tmp = (-1.*temp2) - (re_b[n]*re_Mo1n3 - im_b[n]*im_Mo1n3);
			im_tmp = temp1 - (re_b[n]*im_Mo1n3 + im_b[n]*re_Mo1n3);
			re_Es3[m] += (re_En * re_tmp) - (im_En * im_tmp);
			im_Es3[m] += (im_En * re_tmp) + (re_En * im_tmp);

			pi_nm1[m] = pi_n[m];
			pi_n[m] = swisc[m] + (n+1.)*twisc[m]/n;
			re_xi_nm2[m] = re_xi_nm1[m];
			im_xi_nm2[m] = im_xi_nm1[m];
			re_xi_nm1[m] = re_xi_n[m];
			im_xi_nm1[m] = im_xi_n[m];

		}

		if (powtemp==1)
			powtemp++;
		else if (powtemp==2)
			powtemp++;
		else if (powtemp==3)
			powtemp++;
		else if (powtemp==4)
			powtemp=1;
	}

	for (m=sizem; m--; )
	{
		//Es1 is radial
		//Es2 is theta direction
		//Es3 is phi direction

		//These three "temp"s are used for the Fresnel coefficients (used
		//for correcting the wave passing through planar interfaces)
		temp = sqrt(1-pow(sa_coeff*sintheta[m],2));

		temp1 = 2*costhetanf[m] / (temp + (sa_coeff*costheta[m]));
	
		temp2 = 2*costhetanf[m] / (costheta[m] + (sa_coeff*temp));

		re_Es1[m] *= cosphi[m]*sintheta[m]/pow(kr[m],2);
		im_Es1[m] *= cosphi[m]*sintheta[m]/pow(kr[m],2);
		re_Es2[m] *= cosphi[m]/kr[m];
		im_Es2[m] *= cosphi[m]/kr[m];
		re_Es3[m] *= sinphi[m]/kr[m];
		im_Es3[m] *= sinphi[m]/kr[m];

		// convert to cartesian coordinates
		re_Ec1[m] = re_Es1[m] * sintheta[m] * cosphi[m];
		re_Ec1[m] += re_Es2[m] * costheta[m] * cosphi[m];
		re_Ec1[m] -= re_Es3[m] * sinphi[m];
		im_Ec1[m] = im_Es1[m] * sintheta[m] * cosphi[m];
		im_Ec1[m] += im_Es2[m] * costheta[m] * cosphi[m];
		im_Ec1[m] -= im_Es3[m] * sinphi[m];

		//re_Ec1[m] /= temp1;
		//im_Ec1[m] /= temp1;

		re_Ec2[m] = re_Es1[m] * sintheta[m] * sinphi[m];
		re_Ec2[m] += re_Es2[m] * costheta[m] * sinphi[m];
		re_Ec2[m] += re_Es3[m] * cosphi[m];
		im_Ec2[m] = im_Es1[m] * sintheta[m] * sinphi[m];
		im_Ec2[m] += im_Es2[m] * costheta[m] * sinphi[m];
		im_Ec2[m] += im_Es3[m] * cosphi[m];

		//re_Ec2[m] /= temp2;
		//im_Ec2[m] /= temp2;

		re_Ec3[m] = re_Es1[m]*costheta[m] - re_Es2[m]*sintheta[m];
		im_Ec3[m] = im_Es1[m]*costheta[m] - im_Es2[m]*sintheta[m];

	}

	free(xs);			//1
	free(ys);			//2
	free(pi_nm1);		//3
	free(pi_n);			//4
	free(re_xi_nm2);	//5
	free(im_xi_nm2);	//6
	free(re_xi_nm1);	//7
	free(im_xi_nm1);	//8
	free(re_xi_n);		//9
	free(im_xi_n);		//10
	free(cosphi);		//11
	free(sinphi);		//12
	free(sintheta);		//13
	free(costheta);		//14
	free(kr);			//15
	free(re_Es1);		//16
	free(im_Es1);		//17
	free(re_Es2);		//18
	free(im_Es2);		//19
	free(re_Es3);		//20
	free(im_Es3);		//21
	free(swisc);		//22
	free(twisc);		//23
	return 0;
}


void *scattering_matrix (double *re_S1, double *re_S2,
		   double *im_S1, double *im_S2, 
		   int nc, float xstart, int numx, float xstep,
		   float ystart, int numy, float ystep,
		   float k, float z,
		   float *re_a, float *im_a, 
		   float *re_b, float *im_b)
{
	/*
	Calculates the scattering matrix elements S1 and S2.
	See Bohren and Hoffman eqns 4.74
	This assumes that kr >> nc.
	*/
	float *xs;
	float *ys;
	float rho;
	float *kr;
	double temp, temp1, temp2;
	double re_tmp, im_tmp;
	float *costheta;
	double *pi_nm1;
	double *pi_n;
	double *swisc, *twisc;
	double tau_n;
	unsigned int m=0;
	unsigned int sizem = numx*numy;
	unsigned int i,j,n;

	xs = (float*)malloc(numx*sizeof(float));
	ys = (float*)malloc(numy*sizeof(float));
	kr = (float*)malloc(sizem*sizeof(float));
	costheta = (float*)malloc(sizem*sizeof(float));
	pi_nm1 = (double*)malloc(sizem*sizeof(double));
	pi_n = (double*)malloc(sizem*sizeof(double));
	swisc = (double*)malloc(sizem*sizeof(double));
	twisc = (double*)malloc(sizem*sizeof(double));

	for (i=0; i<numx; i++)
	{
		xs[i] = xstart + (i*xstep);

		for (j=0; j<numy; j++)
		{
			ys[j] = ystart + (j*ystep);

			rho = sqrt(pow(xs[i],2)+pow(ys[j],2));
			kr[m] = k*sqrt(pow(rho,2)+pow(z,2));

			
			//Let temp be theta:
			temp = atan2(rho,z);

			costheta[m] = cos(temp);

			pi_nm1[m] = 0.;
			pi_n[m] = 1.;

			re_S1[m] = 0.;
			re_S2[m] = 0.;
			im_S1[m] = 0.;
			im_S2[m] = 0.;

			m++;
		}
	}

	for (n = 1; n<nc; n++)
	{
		temp = (n + n + 1.) / n / (1.+n);
		for (m=sizem; m--; )
		{
			
			swisc[m] = pi_n[m] * costheta[m];
			twisc[m] = swisc[m] - pi_nm1[m];
			tau_n = (n*twisc[m]) - pi_nm1[m];

			re_S1[m] += temp * (re_a[n] * pi_n[m] + re_b[n] * tau_n);
			im_S1[m] += temp * (im_a[n] * pi_n[m] + im_b[n] * tau_n);
	
			re_S2[m] += temp * (re_a[n] * tau_n + re_b[n] * pi_n[m]);
			im_S2[m] += temp * (im_a[n] * tau_n + im_b[n] * pi_n[m]);

			pi_nm1[m] = pi_n[m];
			pi_n[m] = swisc[m] + (n+1.)*twisc[m]/n;

		}

	}

	free(xs);			//1
	free(ys);			//2
	free(pi_nm1);		//3
	free(pi_n);			//4
	free(costheta);		//5
	free(kr);			//6
	free(swisc);		//7
	free(twisc);		//8
	return 0;
}



void miehol(double *result, double z, float a,
			float nm,
			int xdim, int ydim,
			double mpp,
			float lam,
			double alpha,
			double *re_Ec1, double *im_Ec1, double xs, double ys)
{
	//float alpha = 0.1;
	float lam_in_media = lam / nm;
	float rho2, tmp2;
	float size_param = 2. * 3.14159265 * a / lam_in_media;
	float *realpart, *imagpart;
	unsigned int sizem = xdim*ydim;
	double re_tmp, im_tmp, tmp;
	double sum_fld = 0.;
	unsigned int m = 0;
	unsigned int i,j;
	unsigned int k=0;
	double min, max;
	double weight;
	float xstep=1;
	float ystep=1;
	double correctz;

	realpart = (float*)malloc(sizem*sizeof(float));
	imagpart = (float*)malloc(sizem*sizeof(float));

	tmp = 2. * 3.14159265 * z / lam_in_media;
	re_tmp = cos(tmp);
	im_tmp = -1. * sin(tmp);
	
	for (i = 0; i<xdim; i++)
	{
		for (j=0; j<xdim; j++)
		{
			if (planecorrect) {
				rho2 = sqrt(pow(xs + (i*xstep),2) + pow(ys + (j*ystep),2));
				//correctz = (z/glob_mpp)-(sa_param*pow(rho2,2));
				//tmp = 2. * 3.14159265 * (correctz * glob_mpp) / lam_in_media;
				//re_tmp = cos(tmp);
				//im_tmp = -1. * sin(tmp);
				tmp2 = sa_param*pow(rho2*mpp,4);
				re_Ec1[m] = re_Ec1[m]*cos(tmp2) - im_Ec1[m]*sin(tmp2);
				im_Ec1[m] = re_Ec1[m]*sin(tmp2) + im_Ec1[m]*cos(tmp2);
				}
			realpart[m] = re_Ec1[m]*re_tmp - im_Ec1[m]*im_tmp;
			imagpart[m] = re_Ec1[m]*im_tmp + im_Ec1[m]*re_tmp;
			//sum_fld += pow(realpart[m],2) + pow(imagpart[m],2);
			sum_fld = pow(realpart[m],2) + pow(imagpart[m],2);
			result[m] = 1. + 2.*alpha*realpart[m] + pow(alpha,2)*sum_fld;
			m++;
		}
	}

	//Apply weighting to extremes of hologram to get those
	//higher order fringes
	if (to_weight)
	{
		for (i=0; i<xdim; i++)
		{
			for(j=0; j<ydim; j++)
			{
				weight = pow((float)(i-(xdim/2.)),2) + pow((float)(j-(ydim/2.)),2);
				weight = 1 + (weight / 12500.);
				//result[k] *= weight;
				result[k] = 1+((result[k]-1)*weight);
				k++;
			}
		}
	}

	free(realpart);
	free(imagpart);
}

void miehol_dimer(double *result, double z, double z2, float a,
				  float nm, int xdim, int ydim, double mpp,
				  float lam, double alpha, 
				  double *re_Ec1, double *im_Ec1,
				  double *re_Ec1_2, double *im_Ec1_2,
				  double *re_Ec2, double *im_Ec2,
				  double *re_Ec2_2, double *im_Ec2_2)
{
	float lam_in_media = lam / nm;
	float rho2;
	float size_param = 2. * 3.14159265 * a / lam_in_media;
	float *realpart, *imagpart;
	unsigned int sizem = xdim*ydim;
	double re_tmp, im_tmp, tmp;
	double re_tmp2, im_tmp2, tmp2;
	double sum_fld = 0.;
	unsigned int m = 0;
	unsigned int i,j;
	unsigned int k=0;
	double min, max;
	double weight;
	float xstep=1;
	float ystep=1;
	double correctz;
	double cross1, cross2;
	double tmp3, re_tmp3, im_tmp3;

	realpart = (float*)malloc(sizem*sizeof(float));
	imagpart = (float*)malloc(sizem*sizeof(float));

	tmp = 2. * 3.14159265 * z / lam_in_media;
	re_tmp = cos(tmp);
	im_tmp = -1. * sin(tmp);

	tmp2 = 2. * 3.14159265 * z2 / lam_in_media;
	re_tmp2 = cos(tmp2);
	im_tmp2 = -1. * sin(tmp2);
	tmp3 = 2. * 3.14159265 * (z-z2) / lam_in_media;
	re_tmp3 = cos(tmp3);
	im_tmp3 = -1.0*sin(tmp3);
	
	for (i = 0; i<xdim; i++)
	{
		for (j=0; j<xdim; j++)
		{
			realpart[m] = re_Ec1[m]*re_tmp - im_Ec1[m]*im_tmp;
			realpart[m] += re_Ec1_2[m]*re_tmp2 - im_Ec1_2[m]*im_tmp2;
			imagpart[m] = re_Ec1[m]*im_tmp + im_Ec1[m]*re_tmp;
			imagpart[m] += re_Ec1_2[m]*im_tmp2 + im_Ec1_2[m]*re_tmp2;

			cross1 = re_tmp3 * (re_Ec1[m]*re_Ec1_2[m] + im_Ec1[m]*im_Ec1_2[m]);
			cross1 += im_tmp3 * (re_Ec1[m]*im_Ec1_2[m] - im_Ec1[m]*re_Ec1_2[m]);
			cross1 += re_tmp3 * (re_Ec2[m]*re_Ec2_2[m] + im_Ec2[m]*im_Ec2_2[m]);
			cross1 += im_tmp3 * (re_Ec2[m]*im_Ec2_2[m] - im_Ec2[m]*re_Ec2_2[m]);

			sum_fld = pow(realpart[m],2) + pow(imagpart[m],2);
			result[m] = 1. + 2.*alpha*realpart[m] + pow(alpha,2)*(sum_fld+(2*cross1));
			m++;
		}
	}

	free(realpart);
	free(imagpart);
}

void outputhol (double xstart, double ystart, double zparam, double re_np,
		double re_nm, double radius, double alpha, int x_dim, 
		int y_dim, double mpp, float wavelength, char* fn)
		/*This function outputs a hologram to a file. The filename is given
		by the last argument. File saved as text with \n separating values.*/
{
	int nc;
	int n = x_dim*y_dim;
	float *ASpsi;
	float *ASeta;
	float *re_dns;
	float *im_dns;
	float *re_a={0};
	float *im_a={0};
	float *re_b={0};
	float *im_b={0};
	double z = zparam*1e-6;
	double re_m;
	double im_m = imm_glob;
	double sizeparam;
	double *re_Ec1={0}, *re_Ec2={0}, *re_Ec3={0};
	double *im_Ec1={0}, *im_Ec2={0}, *im_Ec3={0};
	unsigned int i=0;
	double z_in_pixels, k_in_pixels;
	double *holdata;

	FILE *outp = fopen(fn, "w");

	sizeparam = (2*3.14159265*re_nm*radius)/(1e6 * wavelength);
	re_m = re_np / re_nm;
    
    nterms = nstop(sizeparam);

	re_Ec1 = (double*)malloc(n*sizeof(double));
	re_Ec2 = (double*)malloc(n*sizeof(double));
	re_Ec3 = (double*)malloc(n*sizeof(double));
	im_Ec1 = (double*)malloc(n*sizeof(double));
	im_Ec2 = (double*)malloc(n*sizeof(double));
	im_Ec3 = (double*)malloc(n*sizeof(double));
	ASpsi = (float*)malloc(nterms*sizeof(float));
	ASeta = (float*)malloc(nterms*sizeof(float));
	re_dns = (float*)malloc(nterms*sizeof(float));
	im_dns = (float*)malloc(nterms*sizeof(float));
	re_a = (float*)malloc(nterms*sizeof(float));
	im_a = (float*)malloc(nterms*sizeof(float));
	re_b = (float*)malloc(nterms*sizeof(float));
	im_b = (float*)malloc(nterms*sizeof(float));
	holdata = (double*)malloc(n*sizeof(double));

	for (i=0; i<nterms; i++)
	{
		ASpsi[i]=0;
		ASeta[i]=0;
		re_a[i]=0;
		im_a[i]=0;
		re_b[i]=0;
		im_b[i]=0;
	}

	nc = nstop(sizeparam);
	Dn2( nc, sizeparam, re_m, im_m, &re_dns[0], &im_dns[0]);
	AS(nc,sizeparam,&ASpsi[0],&ASeta[0]);

	for (i=0; i<nc; i++)
	{
		im_dns[i] *= -1;
	}
	z_in_pixels = z*1e6/mpp;
	k_in_pixels = (2 * 3.14159265) / (wavelength / re_nm / (mpp*1e-6));
	a_and_b(sizeparam,re_m,im_m,nterms,&re_a[0],&im_a[0],&re_b[0],&im_b[0],&ASpsi[0],&ASeta[0],&re_dns[0],&im_dns[0]);

	flds(&re_Ec1[0], &re_Ec2[0], &re_Ec3[0], &im_Ec1[0], &im_Ec2[0], &im_Ec3[0], nterms,xstart,x_dim,1,ystart,y_dim,1,k_in_pixels,z_in_pixels,&re_a[0],&im_a[0],&re_b[0],&im_b[0]);

	/*
	if ((xs_sa == 0) && (ys_sa == 0))
		miehol(holdata, z, 1e-6, re_nm, x_dim, y_dim, wavelength, alpha, &re_Ec1[0], &im_Ec1[0], xstart, ystart);
	else
		miehol(holdata, z, 1e-6, re_nm, x_dim, y_dim, wavelength, alpha, &re_Ec1[0], &im_Ec1[0], xs_sa, ys_sa);
	*/

	miehol(holdata, z, 1e-6, re_nm, x_dim, y_dim, mpp, wavelength, alpha, &re_Ec1[0], &im_Ec1[0], 0, 0);

	for (i=0; i<n; i++)
	{
		fprintf(outp, "%.15f\n", holdata[i]);
	}

	fclose(outp);


	free(re_Ec1);
	free(im_Ec1);
	free(re_Ec2);
	free(im_Ec2);
	free(re_Ec3);
	free(im_Ec3);
	free(ASpsi);
	free(ASeta);
	free(re_dns);
	free(im_dns);
	free(re_a);
	free(im_a);
	free(re_b);
	free(im_b);
	free(holdata);

}


void *read_hol_bin(double *holdata, char *fn, int x_dim, int y_dim)
{
	int i,j;
	int k=0;
	double weight=0;
	//FILE *outp = fopen("weighted_input.dat","w");


	FILE *holfile = fopen(fn,"r"); //data file
	if (holfile == NULL)
	{
		printf("Couldn't open file.\n");
	}
	fread(holdata, sizeof(double), x_dim*y_dim, holfile);
	fclose(holfile);

	//Apply weighting to extremes of hologram to get those
	//higher order fringes
	if (to_weight)
	{
		for (i=0; i<x_dim; i++)
		{
			for(j=0; j<y_dim; j++)
			{
				weight = pow((float)(i-(x_dim/2.)),2) + pow((float)(j-(y_dim/2.)),2);
				weight = 1 + (weight / 12500.);
				//holdata[k] *= weight;
				holdata[k] = 1+((holdata[k]-1)*weight);
				k++;
			}
		}
	}

	//This would output the weighted hologram.
	//Just to see what it'd look like (testing purposes really).
	/*
	for (i=0; i<(x_dim*y_dim); i++)
	{
		fprintf(outp, "%.15f\n", holdata[i]);
	}
	fclose(outp);
	*/

	return 0;
}

float hologramFitError (double xstart, double ystart, double zparam, double re_np,
		double re_nm, double radius, double alpha, int n, int x_dim, int y_dim, 
		double mpp, float wavelength, char* input_data_fn)
{
	int nc;
	float *ASpsi;
	float *ASeta;
	float *re_dns;
	float *im_dns;
	float *re_a={0};
	float *im_a={0};
	float *re_b={0};
	float *im_b={0};
	double z = zparam*1e-6;
	double re_m;
	double im_m = imm_glob;
	double sizeparam;
	double *re_Ec1={0}, *re_Ec2={0}, *re_Ec3={0};
	double *im_Ec1={0}, *im_Ec2={0}, *im_Ec3={0};
	unsigned int i=0;
	double z_in_pixels, k_in_pixels;
	double *holdata;
	double *input_hologram;
	float error = 0;

	nterms = nstop(sizeparam);

	sizeparam = (2*3.14159265*re_nm*radius)/(1e6 * wavelength);
	re_m = re_np / re_nm;

	re_Ec1 = (double*)malloc(n*sizeof(double));
	re_Ec2 = (double*)malloc(n*sizeof(double));
	re_Ec3 = (double*)malloc(n*sizeof(double));
	im_Ec1 = (double*)malloc(n*sizeof(double));
	im_Ec2 = (double*)malloc(n*sizeof(double));
	im_Ec3 = (double*)malloc(n*sizeof(double));
	ASpsi = (float*)malloc(nterms*sizeof(float));
	ASeta = (float*)malloc(nterms*sizeof(float));
	re_dns = (float*)malloc(nterms*sizeof(float));
	im_dns = (float*)malloc(nterms*sizeof(float));
	re_a = (float*)malloc(nterms*sizeof(float));
	im_a = (float*)malloc(nterms*sizeof(float));
	re_b = (float*)malloc(nterms*sizeof(float));
	im_b = (float*)malloc(nterms*sizeof(float));
	holdata = (double*)malloc(n*sizeof(double));

	input_hologram = (double*)malloc(n*sizeof(double));	

	read_hol_bin(input_hologram, input_data_fn, x_dim, y_dim);

	for (i=0; i<nterms; i++)
	{
		ASpsi[i]=0;
		ASeta[i]=0;
		re_a[i]=0;
		im_a[i]=0;
		re_b[i]=0;
		im_b[i]=0;
	}

	nc = nstop(sizeparam);

	Dn2( nc, sizeparam, re_m, im_m, &re_dns[0], &im_dns[0]);
	AS(nc,sizeparam,&ASpsi[0],&ASeta[0]);

	for (i=0; i<nterms; i++)
	{
		im_dns[i] *= -1;
	}
	z_in_pixels = z*1e6/mpp;
	k_in_pixels = (2 * 3.14159265) / (wavelength / re_nm / (mpp*1e-6));
	a_and_b(sizeparam,re_m,im_m,nc,&re_a[0],&im_a[0],&re_b[0],&im_b[0],&ASpsi[0],&ASeta[0],&re_dns[0],&im_dns[0]);

	flds(&re_Ec1[0], &re_Ec2[0], &re_Ec3[0], &im_Ec1[0], &im_Ec2[0], &im_Ec3[0], nc,xstart,x_dim,1,ystart,y_dim,1,k_in_pixels,z_in_pixels,&re_a[0],&im_a[0],&re_b[0],&im_b[0]);

	/*
	if ((xs_sa == 0) && (ys_sa == 0))
		miehol(holdata, z, 1e-6, re_nm, x_dim, y_dim, wavelength, alpha, &re_Ec1[0], &im_Ec1[0], xstart, ystart);
	else
		miehol(holdata, z, 1e-6, re_nm, x_dim, y_dim, wavelength, alpha, &re_Ec1[0], &im_Ec1[0], xs_sa, ys_sa);
	*/
	miehol(holdata, z, 1e-6, re_nm, x_dim, y_dim, mpp, wavelength, alpha, &re_Ec1[0], &im_Ec1[0], 0, 0);


	for (i=0; i<n; i++)
	{
		error += pow(holdata[i] - input_hologram[i], 2);
	}

	free(re_Ec1);
	free(im_Ec1);
	free(re_Ec2);
	free(im_Ec2);
	free(re_Ec3);
	free(im_Ec3);
	free(ASpsi);
	free(ASeta);
	free(re_dns);
	free(im_dns);
	free(re_a);
	free(im_a);
	free(re_b);
	free(im_b);
	free(holdata);
	free(input_hologram);

	return error;

}
