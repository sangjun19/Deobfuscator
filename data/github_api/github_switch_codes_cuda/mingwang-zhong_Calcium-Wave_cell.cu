// Repository: mingwang-zhong/Calcium-Wave
// File: intact/Hill7_cstar1.1_ISO_NoATP/cell.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
using namespace std;

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
// #define Permeabilized		// Permeabalized cell. No sarcolemmal ion channels
// #define PermeabilizedB		// Permeabalized cell. Myocyte diffuses with a bath ( boundary condition )
// #define LQT2			// Long-QT 2 syndrome simulation. No I_Kr
#define ISO			// Isoproterenol, increases Uptake and I_Ca,L (and/or IKs)
// #define Vclamp		// step function voltage clamp
// #define APclamp		// action potential clamp

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
#define DT 	0.02 //ms, time step
#define stoptime (25001.0) //ms
#define PCL 2000.0	//ms, pacing cycle length
#define stopbeat 10
#define time_before_beat 100 //ms

#define out_step	50 // number of steps to output data
// #define output_linescan // output linescan
// #define output_fluoxt // output fluorescence

#ifdef	output_fluoxt
	#include <fftw3.h>
#endif
//////////////////////////////////////////////////////////////////
////////////////// CUDA block size 

#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 7
#define BLOCK_SIZE_Z 4
#define Nx 	64	// Number of CRUs in the x direction
#define Ny 	28	// Number of CRUs in the y direction
#define Nz 	12	// Number of CRUs in the z direction
#define Nix	2	// number of lattices in x direction in a CRU
#define Niy	2	// number of lattices in y direction in a CRU
#define Niz	2	// number of lattices in z direction in a CRU
#define Nci	8 //(Nix*Niy*Niz) // number of lattices in a CRU
#define DX	(1.65/Nix) // um, size of each lattice in longitudinal direction
#define DY	(0.76/Niy) // um
#define DZ	(0.76/Niz) // um

//////////////////////////////////////////////////////////////////
////////////////// cell properties

#define Vp 0.00126 	//um^3, Volume of the proximal space
#define Vs 0.025 //um^3, Volume of the submembrane space
#define Vjsr 0.03	//um^3, Volume of the JSR space
#define Vi (0.76/Nci)	//um^3, Volume of the cytosolic space, for each compartment
#define Vnsr (0.025/Nci)	//um^3, Volume of the NSR space
#define taups 0.01	//ms, Diffusion time from the proximal to the submembrane
#define taupi 0.05 //ms, Diffusion time from the proximal to the cytosol
#define tausi 0.02		//ms, Diffusion time from the submembrane to the cytosolic
#define taust 1.5  //ms, diffusion time in submembrane along transverse direction
#define tautr 4.25	//ms, Diffusion time from NSR to JSR 
#define taunl 4.2		//ms, diffusion time of longitudinal NSR
#define taunt 1.3		//ms, diffusion time of transverse NSR
// #define tauil 2.7	//ms, diffusion time of longitudinal cytosolic
#define tauit 0.963 	//ms, diffusion time of transverse cytosolic
#define	Datp 	(5.8/2.7)	// diffusion time prefactor of ATP
#define Ddye 	(5.4/2.7)	// diffusion time prefactor of dye 0.09*765/1100
#define Degta 	(7.0/2.7)	// diffusion time prefactor of EGTA 

#define ci_basal (atof(argv[2]))
#define cjsr_basal (atof(argv[3]))
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
//////////// ion channel parameters 

#define	gKs  (0.4)  // mS/uF, IKs conductance
#define gtof (0.1)  // mS/uF, Itof conductance
#define gtos (0.04)  // mS/uF, Itos conductance
#define gNaK (1.5)  // mS/uF, INaK conductance
#define gK1  (0.3)  // mS/uF, IK1 conductance
#define gNa  (12.0)  // mS/uF, INa conductance
#define gNaLeak (0.0015)  // mS/uF, INaLeak conductance
#define Vncx 	(25.0) //uM/ms, strength of NCX current
#define	Vleak (0.00212) // ms^{-1}, Shannon et al 2004, Eq. 107 (0.00212 = 5.348e-6*0.5/0.00126 )

#define FracSL 0.89 // distribution between submembrane and dayad: 0.89:0.11

#ifdef ISO
	#define Vup (200.0) // uM/ms, uptake strength
#else
	#define Vup (140.0) // uM/ms
#endif

#ifdef LQT2
	#define	gKr 0
#else
	#define	gKr 0.024  // mS/uF, IKs conductance
#endif

// ryr gating
#define nryr 	84		//Number of Ryr channels
// #define f_Jmax 	18.0	//Jmax prefactor
#define taub 1.0 	    // ms, transition rate from CSQN-unbound states to CSQN-bound states
#define taucu 2.5    // ms, transition rate from open-unbound state to closed-unbound state
#define taucb 2.5    // ms, transition rate from open-bound state to closed-bound state
#define corbularSR_percentage	0.05
#define Spark_Threshold 3000.0 // uM/ms, when RyR release flux is larger than this, it is a spark.
#define Sparks_Interval 100.0 // ms, minimum time interval between sparks

// luminal gating
#define nCa 31.0 // number of Ca2+ binding sites of each CSQN molecule
#define BCSQN	460.0 //uM, concentration of CSQN
#define Kc 600.0 //uM, Dissociation constant of CSQN

// LCC ica
#define Pca 11.9	// umol/C/ms, 11.9: Restrepo 2008
#define NLCC 5	 // number of LCC channels in each dyadic space
#define gammai 0.341 // Activity coefficient of Ca2+

// NCX
#define	NaO 	136.0 // mM, [Na+]o
#define Kmcai 	0.00359 // uM
#define Kmcao 	1.3 // mM
#define Kmnai 	12.3 // mM
#define Kmnao 	87.5 // mM
#define eta		0.35
#define ksat	0.27

// other
#define CaO 1.8		//mM, external Ca2+ concentration
#define KI 	140.0	//mM, internal K+ concentration
#define KO 	5.40	//mM, external K+ concentration
#define Ek  ( (1.00/FRT)*log(KO/KI) )	//mV
#define Cm 	45 // pF, Capacitance of the whole cell membrane
#define Faraday 96.485		//	C/mmol
#define RR	8.314			//	J/mol/K
#define Temperature	308		//	K
#define FRT (Faraday/RR/Temperature)
#define PI 	3.1415926

#define pos(x,y,z)		(Nx*Ny*(z)+Nx*(y)+(x))
#define posi(i,j,k)		(Nix*Niy*(k)+Nix*(j)+(i))	// position in a CRU
#define posall(i,j,k)	((k)*(Nx*Nix)*(Ny*Niy)+(j)*(Nx*Nix)+(i))	// whole cell position
#define posallf(i,j,k)	((k)*(Nx*Nix/2+1)*Ny*Niy+(j)*(Nx*Nix/2+1)+(i))// whole cell position in k space(FFT)

#define pow2(x) ((x)*(x))
#define pow3(x) ((x)*(x)*(x))
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

#define ktson 	0.00254 // (uM*ms)^(-1), associate constant
#define ktsoff 	0.000033 // ms^(-1), dissociate constant
#define Bts 	134.0	//uM, Mg binding/unbinding does not occur very much. Bts = steady state free+cabound troponin.
#define ktfon 	0.0327
#define ktfoff 	0.0196 // Troponin fast
#define Btf 	70.0

#define kcalon 	0.0543 // Calmodulin
#define kcaloff 0.238
#define Bcal 	24.0

#define ksron 	0.1 // SR
#define ksroff 	0.06
#define Bsr 	47.0

#define ksaron 	0.1 // Sarcolemma
#define ksaroff 1.3
#define Bsar 	(42*(Vi*Nci/Vs)) // in Bers book, it is 42 uM/l cytosol

#define ksarhon 	0.1  // Membrane/High
#define ksarhoff 	0.03
#define Bsarh 		(15.0*(Vi*Nci/Vs))

#define Bmyo 	140.0 // Myosin
#define konmyomg 	0.0000157
#define koffmyomg 	0.000057
#define konmyoca 	0.0138
#define koffmyoca 	0.00046
#define Mgi 	1000.0 // Mg
#define Kmyomg 	(koffmyomg/konmyomg)
#define Kmyoca 	(koffmyoca/konmyoca)

#define katpon 	0.15 // ATP
#define katpoff 30.0
#define Batp 	0//727.0

#define kdyeon		0.08 // Dye
#define kdyeoff		0.09
#define Bdye		50.0

#define kegtaon 	0.0015 // EGTA
#define kegtaoff	0.0003
#define Begta		0//350.0

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

// point spread function for fluoxt.txt
#define SX	(1.44/sqrt(8*log(2))) // sigma of the point spread function
#define SY	(0.47/sqrt(8*log(2)))
#define SZ	(0.47/sqrt(8*log(2)))
#define gx(i,ix,Ax) ( 1.0/sqrt(2*PI*SX*SX)*exp(-((i*Nix+ix)-(Ax))*((i*Nix+ix)-(Ax))*DX*DX/(2.0*SX*SX)) )
#define gy(j,jy,Ay)	( 1.0/sqrt(2*PI*SY*SY)*exp(-((j*Niy+jy)-(Ay))*((j*Niy+jy)-(Ay))*DY*DY/(2.0*SY*SY)) )
#define gz(k,kz,Az)	( 1.0/sqrt(2*PI*SZ*SZ)*exp(-((k*Niz+kz)-(Az))*((k*Niz+kz)-(Az))*DZ*DZ/(2.0*SZ*SZ))	)
#define G(i,j,k,ix,jy,kz,Ax,Ay,Az)	( gx(i,ix,Ax) * gy(j,jy,Ay) * gz(k,kz,Az) )

struct sl_bu{
	double casar; // uM, Ca bound Sarcolemma buffer concentration in submembrane space
	double casarh; // uM, Ca bound Membrane/High
	double caatp; // uM, Ca bound ATP
	double cadye; // uM, Ca bound Dye
	double caegta; // uM, Ca bound EGTA
	double caatpnext; // uM, Ca bound ATP
	double cadyenext; // uM, Ca bound Dye
	double caegtanext; // uM, Ca bound EGTA

	double casarj; // uM, Ca bound Sarcolemma in dyad
	double casarhj; // uM, Ca bound Membrane/High in dyad
	double caatpj; // uM, Ca bound ATP in dyad
	double cadyej; // uM, Ca bound Dye in dyad
	double caegtaj; // uM, Ca bound EGTA in dyad
	double caatpjnext; // uM, Ca bound ATP in dyad
	double cadyejnext; // uM, Ca bound Dye in dyad
	double caegtajnext; // uM, Ca bound EGTA in dyad
};

struct cyt_bu{ // cytosolic buffers
	double cacal;
	double catf;
	double cats;
	double casr;
	double camyo;
	double mgmyo;
	double caatp;
	double cadye;
	double caegta;
	double caatpnext;
	double cadyenext;
	double caegtanext;
};

struct cytosol{
	double Juptake; // uM/ms, SERCA uptake flux
	double ci; // uM
	double cinext;
	double cnsr;
	double cnsrnext;
};

struct cru{
	double JNCX; // uM/ms, NCX flux
	double JCa; // uM/ms, ICa flux
	double Jbg; // uM/ms, sarcolemma background current flux
	double cs;
	double csnext;
};

struct cru2{
	double cp;
	double cpnext;
	double cjsr;
	double Tcj; // uM, total Ca2+ in jSR

	int lcc[NLCC];
	int nLCC_open;

	double Jrel; // uM/ms, SR release flux via RyRs
	double Jleak; // uM/ms, leak flux from JSR to dyad
	int nou;
	int ncu;
	int nob;
	int ncb;

	double tauil;
	double tausl;

	double Ka;	// for Inaca

	double cjsrc; // corbular SR
	int nouc;
	int ncuc;
	int nobc;
	int ncbc;
	double Jrelc;
	double Tcjc;

	curandState state;
};

__global__ void	setup_kernel(unsigned long long seed,cru2 *CRU2);
__global__ void Initial( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double ci_b, double cj_b);
__global__ void Compute( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double v, int step, double nai, 
						double f_Jmax, double alpha, double beta, int Hill, double custar, double cbstar);
__global__ void Finish( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU );

__device__ int ryrgating (double cp, double cjsr, curandState *state, int *ncu, int *nou, int *ncb, int *nob, int i, int j, int k, int step,
							double f_Jmax, double alpha, double beta, int Hill, double custar, double cbstar);
__device__ int number_RyR_transit(curandState *state, int NN, double probability, int upBound);
__device__ int LCCgating(double v, double cp, curandState *state, int i );
__device__ double Single_LCC_Current(double v, double cp); // cp in mM
__device__ double ncx(double v, double cs, double nai, double *Ka);
__device__ double uptake(double ci, double cnsr);

double Ina( double v, double *hh, double *jj, double *mm, double nai );
double Ikr( double v, double *Xkr );
double Iks( double v, double *Xs1, double *Xs2, double *Qks, double cst, double nai );
double Ik1( double v );
double Itos(double v, double *Xtos, double *Ytos);
double Itof(double v, double *Xtof, double *Ytof);
double Inak( double v, double nai );
double sodium(double v, double nai, double I_Na, double I_NaK, double I_NCX);

#ifdef output_fluoxt
	int convolve(double*fluo, cyt_bu *CBU);
#endif

void matrix2file(cytosol *CYT, int step);

int main(int argc, char **argv)
{
	int CudaDevice = 0;	
	if( argc >= 1 ) 
		CudaDevice = atoi(argv[1]);
	cudaSetDevice(CudaDevice);

	size_t ArraySize_cru = Nx*Ny*Nz*sizeof(cru);		// CRU
	size_t ArraySize_cru2= Nx*Ny*Nz*sizeof(cru2);
	size_t ArraySize_cyt = Nci*Nx*Ny*Nz*sizeof(cytosol);	// cytosol space, Nci=Nci
	size_t ArraySize_cbu = Nci*Nx*Ny*Nz*sizeof(cyt_bu);	// cytosol space for the buffers
	size_t ArraySize_sbu = Nx*Ny*Nz*sizeof(sl_bu);		// submembrane space for buffers
	size_t ArraySize_dos = Nx*Ny*Nz*sizeof(double);		// total size of submembrane lattices
	size_t ArraySize_dol = Nci*Nx*Ny*Nz*sizeof(double);	// total # of cytosol lattices

	// Allocate arrays memory in CPU 
	cru *h_CRU;
	cru2 *h_CRU2;
	cytosol *h_CYT;
	cyt_bu *h_CBU;
	sl_bu *h_SBU;
	double *spark_clock, *fluo;
	
	h_CRU = (cru*) malloc(ArraySize_cru);
	h_CRU2 = (cru2*) malloc(ArraySize_cru2);
	h_CYT = (cytosol*) malloc(ArraySize_cyt);
	h_CBU = (cyt_bu*) malloc(ArraySize_cbu);
	h_SBU = (sl_bu*) malloc(ArraySize_sbu);
	spark_clock = (double*) malloc(ArraySize_dos);
	fluo =		(double*) malloc(ArraySize_dol);
	
	//Allocate arrays in GPU
	cru *d_CRU;
	cru2 *d_CRU2;
	cytosol *d_CYT;
	cyt_bu *d_CBU;
	sl_bu *d_SBU;

	cudaMalloc((void**)&d_CRU, ArraySize_cru);
	cudaMalloc((void**)&d_CRU2,ArraySize_cru2);
	cudaMalloc((void**)&d_CYT, ArraySize_cyt);
	cudaMalloc((void**)&d_CBU, ArraySize_cbu);
	cudaMalloc((void**)&d_SBU, ArraySize_sbu);

	/////////////////////////////////// variables /////////////////////////////////////////////////
	int step = 0;
	int i, j, k, ix; // i,j,k for CRU index; ix, jy, kz for lattices in each CRU
	double start_time = clock()/(1.0*CLOCKS_PER_SEC),    end_time;

	double nai = 10.0;//73.987/( 1.0 + 6.707*sqrt(PCL/1000.0) );
	double CaExt = 0, TotalCa = 0, TotalCa_before = 0;

	double v = -86.00;	// voltage
	double mm = 0.0010, hh = 1.00, jj = 1.00;	// INa 
	double Xkr = 0.0; // IKr
	double Xs1 = 0.084, Xs2 = Xs1, Qks = 0.2;	// IKs 
	double Xtos = 0.01, Ytos = 1.0;	// Itos
	double Xtof = 0.02, Ytof = 0.8;	// Itof
	double I_NaK = 0, I_Na = 0, I_Kr = 0, I_Ks = 0, I_K1 = 0, I_tos = 0, I_tof = 0, I_Ca = 0, I_NCX = 0, I_bg = 0;

	double cit, cpt, cst, cjsrt, cnsrt ;
	int Nxyz = (Nx-2)*(Ny-2)*(Nz-2), NcorSR = 0;
	
	/////// to calculate spark rate
	double num_spark = 0, spark_rate = 0, spark_rate_history[20] = {0.0};
	for (k=0;k<Nz;k++)
	{
		for (j=0;j<Ny;j++)
		{
			for (i=0;i<Nx;i++)
			{
				spark_clock[pos(i,j,k)] = Sparks_Interval;
			}
		}
	}

	///////////////////////////////////////////// files /////////////////////////////////////////////////

	FILE * wholecell_file = fopen("wholecell.txt","w");
	
	#ifdef output_linescan
		FILE * linescan_file = fopen("linescan.txt","w");
	#endif

	#ifdef output_fluoxt
		FILE * fluoxt_file = fopen("fluoxt.txt","w");
	#endif

	////////////////////////////////////////////////////////////////////////////////////////////////////	
	// Set paramaters for geometry of computation
	dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 numBlocks(Nx/threadsPerBlock.x, Ny/threadsPerBlock.y, Nz/threadsPerBlock.z);

	setup_kernel<<<numBlocks, threadsPerBlock>>>(98,d_CRU2);
	Initial<<<numBlocks, threadsPerBlock>>>(d_CRU, d_CRU2, d_CYT, d_CBU, d_SBU, ci_basal, cjsr_basal);
	cudaMemcpy(h_CRU, d_CRU, ArraySize_cru, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_CRU2, d_CRU2, ArraySize_cru2, cudaMemcpyDeviceToHost);
	for (k = 1; k < Nz-1; k++)
	{
		for (j = 1; j < Ny-1; j++)
		{
			for (i = 1; i < Nx-1; i++) 
			{
				if (h_CRU2[pos(i,j,k)].cjsrc>0)
					++NcorSR;
			}
		}
	}
	printf("NcorSR = %d\n", NcorSR);

	while ( step*DT < stoptime )
	{
		cudaMemcpy(h_CRU, d_CRU, ArraySize_cru, cudaMemcpyDeviceToHost);

		//////////////////////////////// whole cell average: cst, Ica, Inaca, Ibcg ///////////////////////
		
		if ( step%out_step==1 )
		{
			CaExt=0;	// total Ca2+ exchange through the cell membrane
		}

		cst = 0;
		I_Ca = 0;
		I_bg = 0;
		I_NCX = 0;
		
		for (k = 1; k < Nz-1; k++)
		{
			for (j = 1; j < Ny-1; j++)
			{
				for (i = 1; i < Nx-1; i++) 
				{
					cst += h_CRU[pos(i,j,k)].cs;
					I_Ca += h_CRU[pos(i,j,k)].JCa;
					I_bg += h_CRU[pos(i,j,k)].Jbg;
					I_NCX += h_CRU[pos(i,j,k)].JNCX;

					CaExt = CaExt - h_CRU[pos(i,j,k)].JCa*Vp*DT/Nxyz 
							+ h_CRU[pos(i,j,k)].JNCX*Vs*DT/Nxyz;
				}
			}
		}
		cst=cst/Nxyz;

		// Firstly, I convert uM/ms to pA, then I make it divided by capacitance (Cm, pF)
		// So the unit of I_Ca is pA/pF, equal to mV/ms, which is the unit of dv/dt
		I_Ca = I_Ca*0.0965*Vp*2.0 / Cm;
		I_NCX = I_NCX*0.0965*Vs / Cm;
		I_bg = I_bg*0.0965*Vp*2.0 / Cm;

		//////////////////// other ion channels ///////////////////
		I_Na = Ina(v, &hh, &jj, &mm, nai);
		I_Kr = Ikr(v, &Xkr);
		I_Ks = Iks(v, &Xs1, &Xs2, &Qks, cst, nai );
		I_K1 = Ik1(v);
		I_tos = Itos(v, &Xtos, &Ytos);
		I_tof = Itof(v, &Xtof, &Ytof);
		I_NaK = Inak(v, nai);
		//nai = sodium(v, nai, I_Na, I_NaK, I_NCX);

		///////////////////////////	Action Potential ///////////////////////////////////
		double stim = 0;
		if( fmod(step*DT+PCL-time_before_beat,PCL) < 1.0 && step*DT > time_before_beat && step*DT < PCL*stopbeat )
			stim = 80.0;
		double dvh = -( I_Na + I_K1 + I_Kr + I_Ks + I_tos + I_tof + I_NCX + I_Ca + I_NaK + I_bg ) + stim; 
		v += dvh*DT;

		#ifdef Permeabilized
			v = -86;
		#endif
	
		#ifdef Vclamp
			v = -86;
			if( step*DT > time_before_beat && step*DT < time_before_beat+200 )
				v = 0;
		#endif

		#ifdef APclamp
			v = varray[((int)( t/DT+0.1 ))%((int)(PCL/DT+0.1))];
		#endif
		
		///////////////////////////////////////////////////////////////////////////////
		//////////////////////////////// output ///////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////
		if ( step%out_step==0 )
		{
			cudaMemcpy(h_CRU2,d_CRU2,ArraySize_cru2,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_CYT, d_CYT, ArraySize_cyt, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_CBU, d_CBU, ArraySize_cbu, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_SBU, d_SBU, ArraySize_sbu, cudaMemcpyDeviceToHost);

		//	matrix2file(h_CYT, step);

			cit = 0;
			cpt = 0;
			cjsrt = 0;
			cnsrt = 0;
			TotalCa = 0;

			double catft=0, catst=0, casrt=0, camyot=0, mgmyot=0, cacalt=0, caatpt=0, cadyet=0, caegtat=0;
			double casart = 0, casarht = 0, caatpst = 0, cadyest = 0, caegtast =0;
			double casarjt = 0, casarhjt = 0, caatpjt = 0, cadyejt = 0, caegtajt =0;
			double Jleakt=0, Juptaket=0, Jrelt=0;
			int nout = 0, nobt = 0, ncut = 0, ncbt = 0;
			double icaflux = 0, ncxflux = 0, ncxfwd = 0, Kat = 0;
			double cjt=0;	// total Ca2+ in jSR
			double nlcc_open = 0, nlcc_flip = 0, nlcc_act = 0, nlcc_v = 0, nlcc_Ca = 0;

			int ps = 0;
			for (k = 1; k < Nz-1; k++)
			{
				for (j = 1; j < Ny-1; j++)
				{
					for (i = 1; i < Nx-1; i++) 
					{	
						ps=pos(i,j,k);

						if ( h_CRU[ps].JNCX < 0 )
							ncxfwd += h_CRU[ps].JNCX;

						icaflux += h_CRU[ps].JCa;
						ncxflux += h_CRU[ps].JNCX;
						Kat += h_CRU2[ps].Ka;
						cpt += h_CRU2[ps].cp;
						cjsrt += ( h_CRU2[ps].cjsr + ( (h_CRU2[ps].cjsrc>0)?(h_CRU2[ps].cjsrc):0 ) );
						cjt += h_CRU2[ps].Tcj + h_CRU2[ps].Tcjc;
						Jrelt += h_CRU2[ps].Jrel;
						Jleakt += h_CRU2[ps].Jleak;
						nout += h_CRU2[ps].nou;
						nobt += h_CRU2[ps].nob;
						ncut += h_CRU2[ps].ncu;
						ncbt += h_CRU2[ps].ncb;

						casart += h_SBU[ps].casar;
						casarht += h_SBU[ps].casarh;
						caatpst += h_SBU[ps].caatp;
						cadyest += h_SBU[ps].cadye;
						caegtast+= h_SBU[ps].caegta;
						casarjt += h_SBU[ps].casarj;
						casarhjt += h_SBU[ps].casarhj;
						caatpjt += h_SBU[ps].caatpj;
						cadyejt += h_SBU[ps].cadyej;
						caegtajt += h_SBU[ps].caegtaj;

						for ( ix = 0; ix < Nci; ++ix )
						{
							cit += h_CYT[ps*Nci+ix].ci/Nci;
							cnsrt += h_CYT[ps*Nci+ix].cnsr/Nci;
							catft += h_CBU[ps*Nci+ix].catf/Nci;
							catst += h_CBU[ps*Nci+ix].cats/Nci;
							casrt += h_CBU[ps*Nci+ix].casr/Nci;
							camyot += h_CBU[ps*Nci+ix].camyo/Nci;
							mgmyot += h_CBU[ps*Nci+ix].mgmyo/Nci;
							cacalt += h_CBU[ps*Nci+ix].cacal/Nci;
							caatpt += h_CBU[ps*Nci+ix].caatp/Nci;
							cadyet += h_CBU[ps*Nci+ix].cadye/Nci;
							caegtat+= h_CBU[ps*Nci+ix].caegta/Nci;
							Juptaket += h_CYT[ps*Nci+ix].Juptake/Nci;
							if( h_CYT[ps*Nci+ix].ci > 50.0 )
							{
								cout << step*DT << " " << i << " " << j << " " << k << " " 
									 << ix << " error! ci=" << h_CYT[ps*Nci+ix].ci << endl;
							}
						}

						nlcc_open += h_CRU2[ps].nLCC_open;
						for( int ll = 0; ll < NLCC; ll++ )
						{
							switch ( h_CRU2[ps].lcc[ll] )
							{
								case 1: ++nlcc_flip; break;
								case 2: ++nlcc_act; break;
								case 3: ++nlcc_flip; ++nlcc_act; break;
								case 4: ++nlcc_v; break;
								case 5: ++nlcc_flip; ++nlcc_v; break;
								case 6: ++nlcc_act; ++nlcc_v; break;
								case 7: ++nlcc_flip; ++nlcc_act; ++nlcc_v; break;
								case 8: ++nlcc_Ca; break;
								case 9: ++nlcc_flip; ++nlcc_Ca; break;
								case 10: ++nlcc_act; ++nlcc_Ca; break;
								case 11: ++nlcc_flip; ++nlcc_act; ++nlcc_Ca; break;
								case 12: ++nlcc_v; ++nlcc_Ca; break;
								case 13: ++nlcc_flip; ++nlcc_v; ++nlcc_Ca; break;
								case 14: ++nlcc_act; ++nlcc_v; ++nlcc_Ca; break;
								case 15: ++nlcc_flip; ++nlcc_act; ++nlcc_v; ++nlcc_Ca; break;
							}
						}
						
						
					}
				}
			}
			
			cit /= Nxyz;
			cpt /= Nxyz;
			cjsrt /= (Nxyz+NcorSR);
			cjt /= Nxyz;
			cnsrt /= Nxyz;
			catft /= Nxyz;
			catst /= Nxyz;
			casrt /= Nxyz;
			camyot /= Nxyz;
			mgmyot /= Nxyz;
			cacalt /= Nxyz;
			cadyet /= Nxyz;
			caatpt /= Nxyz;
			caegtat /= Nxyz;
			Jleakt /= Nxyz;
			Juptaket /= Nxyz;
			Jrelt /= Nxyz;
			ncxflux /= Nxyz;
			ncxfwd /= Nxyz;
			icaflux /= Nxyz;
			Kat /= Nxyz;
			casart /= Nxyz;
			casarht /= Nxyz;
			caatpst /= Nxyz;
			cadyest /= Nxyz;
			caegtast/= Nxyz;
			casarjt /= Nxyz;
			casarhjt /= Nxyz;
			caatpjt /= Nxyz;
			cadyejt /= Nxyz;
			caegtajt /= Nxyz;
			nlcc_open /= Nxyz;
			nlcc_flip /= Nxyz;
			nlcc_act /= Nxyz;
			nlcc_v /= Nxyz;
			nlcc_Ca /= Nxyz;
			
			TotalCa =	( cit+ catft + catst + casrt + camyot + cacalt + caatpt + cadyet + caegtat )*Vi*Nci 
						+ ( cst + casart + casarht + caatpst + cadyest + caegtast )*Vs 
						+ ( cpt + casarjt + casarhjt + caatpjt + cadyejt + caegtajt )*Vp 
						+ cjt*Vjsr
						+ cnsrt*Vnsr*Nci;

			//////////////////////////////////// spark rate /////////////////////////////////
			num_spark = 0.0;
			for (k=0;k<Nz;k++)
			{
				for (j=0;j<Ny;j++)
				{
					for (i=0;i<Nx;i++)
					{
						if (h_CRU2[pos(i,j,k)].Jrel>Spark_Threshold && spark_clock[pos(i,j,k)]>Sparks_Interval)
						{
							num_spark = num_spark + 1.0;
							spark_clock[pos(i,j,k)] = 0.0;
						}
						spark_clock[pos(i,j,k)] += out_step*DT;
					}
				}
			}
			

			// output spark rate for every 20 ms. Within 20 ms, all the spark rate values are equal to the average.
			// Time shifted: 20ms. Example: in range [20ms, 39ms], all values are equal to the average value in the ranage [0,19ms]
			spark_rate_history[(step/out_step)%20] = num_spark*200.0/1.8/((Nx-2)*(Ny-2)*(Nz-2))/(out_step*DT/1000.0);

			if ( step%(20*out_step)==0 )
			{
				spark_rate = 0;

				for (i=0;i<20;i++)
					spark_rate += spark_rate_history[i];

				spark_rate = spark_rate/20.0;
			}

			////////////////////////////// output to screen /////////////////////////////
			if ( step%(100*out_step)==0 )
			{
				end_time=clock()/(1.0*CLOCKS_PER_SEC);	
				printf(	"t=%g\t/ %g\t\ttime = %.1fs = %.1fh\t\tcit = %g\t\tcjsrt = %g\n",
						step*DT, stoptime, 
						end_time-start_time, (end_time-start_time)/3600.0, 
						cit, cjsrt
					  );
			}

			////////////////////////////// whole cell ////////////////////////////////// flag
			fprintf(wholecell_file,	"%g %g %g %g %g " "%g %g %g %g %g "
									"%g %g %g %g %g " "%g %g %g %g %g "
									"%g %g %g %g %g " "%g %g %g %g %g "
									"%g %g %g %g %g " "%g %g %g %g\n",

									step*DT, cit,
									cpt, cst, 
									cjsrt, cnsrt, 
									v, I_NCX,
									I_Ca, Juptaket, 

									nai, I_Ks,
									I_Kr, I_K1,
									I_NaK, I_tos,
									I_tof, I_Na,
									Jleakt, Jrelt, 

									nout/(1.0*Nxyz), nobt/(1.0*Nxyz),
									ncut/(1.0*Nxyz), ncbt/(1.0*Nxyz),
									ncxflux*(Vs/Vp), icaflux, 
									I_bg, Kat, 
									TotalCa - TotalCa_before, CaExt, 

									nlcc_open, nlcc_flip,
									nlcc_act, nlcc_v,
									nlcc_Ca, spark_rate,
									cadyet, caatpt,
									TotalCa
					);

			fflush( wholecell_file );
			TotalCa_before = TotalCa;

			////////////////////////////// Line Scan ////////////////////////////////////	flag
			#ifdef output_linescan
				// if ( step*DT > ( stopbeat - 2 )*PCL && step*DT < ( stopbeat + 2 )*PCL )
				// if ( step*DT>12000 && step*DT<25000 )
				{
					for (i =1; i < Nx-1; i++)
					{
						int k = 4, j = Ny/2;
						ps = pos(i,j,k);
						fprintf(linescan_file, 	"%g %g %g %g %g " "%g %g %g %g %g "
												"%g %g %i %i %i " "%i %i %g %g\n",
												
												step*DT,			(double)i,
												h_CYT[ps*Nci].ci, 	h_CRU2[ps].cp,
												h_CRU[ps].cs, 		h_CRU2[ps].cjsr,
												h_CYT[ps*Nci].cnsr,	h_CRU2[ps].Jrel,
												h_CYT[ps*Nci].Juptake,	h_CRU2[ps].Jleak,

												h_CRU[ps].JCa, 		h_CRU[ps].JNCX,
												h_CRU2[ps].nou, 	h_CRU2[ps].nob, 
												h_CRU2[ps].ncu, 	h_CRU2[ps].ncb, 
												h_CRU2[ps].nLCC_open,  h_CRU[ps].Jbg,
												h_CRU2[ps].cjsrc
											 
							);
					}
					fprintf(linescan_file, "\n");
					fflush(linescan_file);
				}
			#endif
			
			///////////////////////////////////////// fluoxt ////////////////////////////////////
			#ifdef output_fluoxt
				if ( step*DT>12000 && step*DT<25000 )
				{
					convolve(fluo, h_CBU);
					for (i=0;i<Nx;i++)
					{
						for ( ix=0;ix<Nix;ix++)
						{
							fprintf(fluoxt_file,"%g\t%g\t%g\t%g\t%g\n",
									
									step*DT,	(i*Nix+ix)*DX, 
									fluo[posall(i*Nix+ix, Ny/2*Niy+0, 4*Niz+0)], fluo[posall(i*Nix+ix,(Ny/2-3)*Niy+0,6*Niz+0)], 
									fluo[posall(i*Nix+ix,(Ny/2+5)*Niy+0,6*Niz+0)]
									
									);
						}
					}
				
					fprintf(fluoxt_file,"\n");
					fflush( fluoxt_file );
				}
			#endif
			
		}

		double f_Jmax = atof(argv[4]);
		int Hill = atoi(argv[5]);
		double alpha = atof(argv[6]);
		double beta = atof(argv[7]);
		double custar = atof(argv[8]);
		double cbstar = atof(argv[9]);

		Compute<<<numBlocks, threadsPerBlock>>>( d_CRU, d_CRU2, d_CYT, d_CBU, d_SBU, v, step, nai, f_Jmax, alpha, beta, Hill, custar, cbstar);
		Finish<<<numBlocks, threadsPerBlock>>>( d_CRU, d_CRU2, d_CYT, d_CBU, d_SBU);
		step++;
	}

	fclose(wholecell_file);
	
	#ifdef output_linescan
		fclose(linescan_file);
	#endif
	
	#ifdef output_fluoxt
		fclose(fluoxt_file);
	#endif


	cudaFree(d_CYT);
	cudaFree(d_CRU);
	cudaFree(d_CRU2);
	cudaFree(d_SBU);
	cudaFree(d_CBU);
	
	free(h_CYT);
	free(h_CRU);
	free(h_CRU2);
	free(h_SBU);
	free(h_CBU);
	free(fluo);
	free(spark_clock);
	
	return EXIT_SUCCESS;
}

__global__ void Initial( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double ci_b, double cj_b)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int ps = pos(i,j,k);

	curandState localState;
	localState=CRU2[ps].state;
	
	for ( int ix = 0; ix < Nci; ++ix )
	{
		int psi = ps*Nci+ix;
		CYT[psi].ci = ci_b;
		CYT[psi].cnsr = cj_b;
		CYT[psi].cinext = ci_b;
		CYT[psi].cnsrnext = cj_b;

		CBU[psi].catf = ktfon*ci_b*Btf/(ktfon*ci_b+ktfoff);
		CBU[psi].cats = ktson*ci_b*Bts/(ktson*ci_b+ktsoff);
		CBU[psi].cacal= kcalon*ci_b*Bcal/(kcalon*ci_b+kcaloff);
		CBU[psi].casr = ksron*ci_b*Bsr/(ksron*ci_b+ksroff);
		CBU[psi].caatp = katpon*ci_b*Batp/(katpon*ci_b+katpoff);
		CBU[psi].caatpnext = katpon*ci_b*Batp/(katpon*ci_b+katpoff);
		CBU[psi].cadye = kdyeon*ci_b*Bdye/(kdyeon*ci_b+kdyeoff);
		CBU[psi].cadyenext = kdyeon*ci_b*Bdye/(kdyeon*ci_b+kdyeoff);
		CBU[psi].caegta = kegtaon*ci_b*Begta/(kegtaon*ci_b+kegtaoff);
		CBU[psi].caegtanext = kegtaon*ci_b*Begta/(kegtaon*ci_b+kegtaoff);
	
		double ratio = Mgi*Kmyoca/(ci_b*Kmyomg);
		CBU[psi].camyo = ci_b*Bmyo/(Kmyoca+ci_b*(ratio+1.0));
		CBU[psi].mgmyo = CBU[psi].camyo*ratio;
	}
	
	SBU[ps].casar = ksaron*ci_b*Bsar/(ksaron*ci_b+ksaroff);
	SBU[ps].casarh = ksarhon*ci_b*Bsarh/(ksarhon*ci_b+ksarhoff);
	SBU[ps].caatp = katpon*ci_b*Batp/(katpon*ci_b+katpoff);	
	SBU[ps].cadye = kdyeon*ci_b*Bdye/(kdyeon*ci_b+kdyeoff);	
	SBU[ps].caegta= kegtaon*ci_b*Begta/(kegtaon*ci_b+kegtaoff);
	SBU[ps].caatpnext = katpon*ci_b*Batp/(katpon*ci_b+katpoff);	
	SBU[ps].cadyenext = kdyeon*ci_b*Bdye/(kdyeon*ci_b+kdyeoff);	
	SBU[ps].caegtanext= kegtaon*ci_b*Begta/(kegtaon*ci_b+kegtaoff);

	SBU[ps].casarj = ksaron*ci_b*Bsar/(ksaron*ci_b+ksaroff);
	SBU[ps].casarhj= ksarhon*ci_b*Bsarh/(ksarhon*ci_b+ksarhoff);
	SBU[ps].caatpj = katpon*ci_b*Batp/(katpon*ci_b+katpoff);
	SBU[ps].cadyej = kdyeon*ci_b*Bdye/(kdyeon*ci_b+kdyeoff);
	SBU[ps].caegtaj= kegtaon*ci_b*Begta/(kegtaon*ci_b+kegtaoff);
	SBU[ps].caatpjnext = katpon*ci_b*Batp/(katpon*ci_b+katpoff);
	SBU[ps].cadyejnext = kdyeon*ci_b*Bdye/(kdyeon*ci_b+kdyeoff);
	SBU[ps].caegtajnext= kegtaon*ci_b*Begta/(kegtaon*ci_b+kegtaoff);

	CRU[ps].cs = ci_b;
	CRU[ps].csnext = ci_b;
	CRU2[ps].cp = ci_b;
	CRU2[ps].cpnext = ci_b;
	CRU2[ps].cjsr = cj_b;
	CRU2[ps].Tcj = cj_b + BCSQN*nCa*cj_b/( Kc+cj_b );

	CRU[ps].Jbg= 0;
	CRU[ps].JCa = 0;
	CRU[ps].JNCX = 0;
	CRU2[ps].Jrel = 0;

	for(int ll=0; ll<NLCC; ll++)
	{
		CRU2[ps].lcc[ll]=3;
	}

	double cb=BCSQN*nCa*CRU2[ps].cjsr/(Kc+CRU2[ps].cjsr);
	double ku2b = 1.0/( 1.0+pow(cb/BCSQN/13.3, 24) )/taub;
	double kb2u = 1.0/(  4000.0/( 1.0+pow(CRU2[ps].cjsr/670.0, 24) ) + 350.0  );
	double fracbound = 1/(1+kb2u/ku2b);

	double nryr0 = nryr + (curand_uniform_double(&localState)-0.5)*41;
	if (nryr0 < 1)
		nryr0 = 1;
	CRU2[ps].nLCC_open = 0;
	CRU2[ps].ncb = int(fracbound*nryr);
	CRU2[ps].ncu = nryr-int(fracbound*nryr);
	CRU2[ps].nob = 0;
	CRU2[ps].nou = 0;

	CRU2[ps].Ka = 0.025;

	if ( curand_uniform_double(&localState) < corbularSR_percentage )
		CRU2[ps].cjsrc = cj_b;
	else
		CRU2[ps].cjsrc = -1;

	CRU2[ps].ncbc = CRU2[ps].ncb;
	CRU2[ps].ncuc = CRU2[ps].ncu;
	CRU2[ps].nobc = CRU2[ps].nob;
	CRU2[ps].nouc = CRU2[ps].nou;

	if ( k>=3 && k<=7 && ( (j==8 && i>=35 && i<=60) || (j==16 && i>=5 && i<=30) || (j==24 && i>=20 && i<=45)  ) )
	{
		CRU2[ps].tauil = 2.269;
		CRU2[ps].tausl = 1000000.0;
	}
	else 
	{
		CRU2[ps].tauil = 2.269;
		CRU2[ps].tausl = 1000000.0; // remove diffusion
	}

	CRU2[ps].state = localState;
}


#define FINESTEP 10
#define DTF 	(DT/FINESTEP)

__global__ void Compute( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double v, int step, double nai,
						double f_Jmax, double alpha, double beta, int Hill, double custar, double cbstar)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int ps = pos(i,j,k);

	curandState localState;
	localState=CRU2[ps].state;

	if ((i*j*k)!=0 && i<Nx-1 && j<Ny-1 && k<Nz-1)
	{
		//////////////////////////////////////////////////////////////////////
		/////////////////////////////////// ICa //////////////////////////////
		//////////////////////////////////////////////////////////////////////
		#ifndef Permeabilized
		{
			CRU2[ps].nLCC_open = 0;
			for (int  LCC_ichannel=0; LCC_ichannel<NLCC; LCC_ichannel++ )
			{
				CRU2[ps].lcc[LCC_ichannel] = LCCgating(v, CRU2[ps].cp, &localState, CRU2[ps].lcc[LCC_ichannel]);
				if ( CRU2[ps].lcc[LCC_ichannel] == 0 )
				{
					CRU2[ps].nLCC_open++;
				}
			}

			double ica = Single_LCC_Current(v, CRU2[ps].cp/1000.0 );
			
			CRU[ps].JCa = CRU2[ps].nLCC_open * ica;
		}
		#endif

		//////////////////////////////////////////////////////////////////////
		/////////////////////////////// INCX ////////////////////////////////
		//////////////////////////////////////////////////////////////////////
		#ifndef Permeabilized
			CRU[ps].JNCX = ncx( v, CRU[ps].cs, nai, &CRU2[ps].Ka );
		#endif

		//////////////////////////////////////////////////////////////////////
		/////////////////////////////// RyR //////////////////////////////////
		//////////////////////////////////////////////////////////////////////
		int Nryr_Open = ryrgating( CRU2[ps].cp, CRU2[ps].cjsr, &localState, &CRU2[ps].ncu, &CRU2[ps].nou, 
								&CRU2[ps].ncb, &CRU2[ps].nob, i, j, k, step, f_Jmax, alpha, beta, Hill, custar, cbstar );
		CRU2[ps].Jrel = Nryr_Open * f_Jmax*0.00015 * (CRU2[ps].cjsr-CRU2[ps].cp)/Vp;
		
		if ( CRU2[ps].cjsrc > 0 )
		{
			int Nryr_Open_c = ryrgating( CYT[ps*Nci+1].ci, CRU2[ps].cjsrc, &localState, &CRU2[ps].ncuc, &CRU2[ps].nouc, 
									&CRU2[ps].ncbc, &CRU2[ps].nobc, i, j, k, step, f_Jmax, alpha, beta, Hill, custar, cbstar );
			CRU2[ps].Jrelc = Nryr_Open_c * f_Jmax*0.00015 * (CRU2[ps].cjsrc-CYT[ps*Nci+1].ci)/(Vi/Nci);
		}
		else
			CRU2[ps].Jrelc = 0;

		CRU2[ps].Jleak = Vleak * (CRU2[ps].cjsr - CRU2[ps].cp); // Shannon et al 2004, Eq.107
		//////////////////////////////////////////////////////////////////////
		/////////////////////// other currents ////////////////////////////
		//////////////////////////////////////////////////////////////////////

		CRU[ps].Jbg = 0; // 0.005*(v-log(CaO*1000/CRU[ps].cs)/2.0/FRT);

		//////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////		
		//////////////////////////////////////////////////////////////////////
		double diffjn0 = (CRU2[ps].cjsr-CYT[ps*Nci].cnsr)/(tautr*2.0);
		double diffjn1 = (CRU2[ps].cjsr-CYT[ps*Nci+4].cnsr)/(tautr*2.0);
		
		double diffjn1c = (CRU2[ps].cjsrc-CYT[ps*Nci+1].cnsr)/(tautr*2.0);
		double diffjn5c = (CRU2[ps].cjsrc-CYT[ps*Nci+5].cnsr)/(tautr*2.0);
		if ( CRU2[ps].cjsrc < 0 )
		{
			diffjn1c = 0;
			diffjn5c = 0;
		}

		double diffpi0 = (CRU2[ps].cp-CYT[ps*Nci].ci)/(taupi*2.0);
		double diffpi1 = (CRU2[ps].cp-CYT[ps*Nci+4].ci)/(taupi*2.0);
		double diffsi0 = (CRU[ps].cs-CYT[ps*Nci].ci)/(tausi*2.0);
		double diffsi1 = (CRU[ps].cs-CYT[ps*Nci+4].ci)/(tausi*2.0);

		double diffpiatp0 = (SBU[ps].caatpj-CBU[ps*Nci].caatp)/(Datp*taupi*2.0);
		double diffpiatp1 = (SBU[ps].caatpj-CBU[ps*Nci+4].caatp)/(Datp*taupi*2.0);
		double diffpidye0 = (SBU[ps].cadyej-CBU[ps*Nci].cadye)/(Ddye*taupi*2.0);
		double diffpidye1 = (SBU[ps].cadyej-CBU[ps*Nci+4].cadye)/(Ddye*taupi*2.0);
		double diffpiegta0 = (SBU[ps].caegtaj-CBU[ps*Nci].caegta)/(Degta*taupi*2.0);
		double diffpiegta1 = (SBU[ps].caegtaj-CBU[ps*Nci+4].caegta)/(Degta*taupi*2.0);

		double diffsiatp0 = (SBU[ps].caatp-CBU[ps*Nci].caatp)/(Datp*tausi*2.0);
		double diffsiatp1 = (SBU[ps].caatp-CBU[ps*Nci+4].caatp)/(Datp*tausi*2.0);
		double diffsidye0 = (SBU[ps].cadye-CBU[ps*Nci].cadye)/(Ddye*tausi*2.0);
		double diffsidye1 = (SBU[ps].cadye-CBU[ps*Nci+4].cadye)/(Ddye*tausi*2.0);
		double diffsiegta0 = (SBU[ps].caegta-CBU[ps*Nci].caegta)/(Degta*tausi*2.0);
		double diffsiegta1 = (SBU[ps].caegta-CBU[ps*Nci+4].caegta)/(Degta*tausi*2.0);
		

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////// dotci ////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for ( int kz=0; kz<Niz; kz++ )
		{
			for ( int jy=0; jy<Niy; jy++ )
			{
				for ( int ix=0; ix<Nix; ix++ )
				{
					int psi = ps*Nci + posi(ix,jy,kz);
					int crui = posi(ix,jy,kz);

					CYT[psi].Juptake = uptake(CYT[psi].ci, CYT[psi].cnsr);

					double bufftf = ktfon*CYT[psi].ci*(Btf-CBU[psi].catf) - ktfoff*CBU[psi].catf;
					double buffts = ktson*CYT[psi].ci*(Bts-CBU[psi].cats) - ktsoff*CBU[psi].cats;
					double buffcal = kcalon*CYT[psi].ci*(Bcal-CBU[psi].cacal) - kcaloff*CBU[psi].cacal;
					double buffsr = ksron*CYT[psi].ci*(Bsr-CBU[psi].casr) - ksroff*CBU[psi].casr;
					double buffmyo = konmyoca*CYT[psi].ci*(Bmyo-CBU[psi].camyo-CBU[psi].mgmyo)-koffmyoca*CBU[psi].camyo;
					double buffmyomg = konmyomg*Mgi*(Bmyo-CBU[psi].camyo-CBU[psi].mgmyo)-koffmyomg*CBU[psi].mgmyo;
					double buffdye = kdyeon*CYT[psi].ci*(Bdye-CBU[psi].cadye) - kdyeoff*CBU[psi].cadye;
					double buffegta = kegtaon*CYT[psi].ci*(Begta-CBU[psi].caegta) - kegtaoff*CBU[psi].caegta;

					int inext =	  (ix==Nix-1)?( pos(i+1,j,k)*Nci+posi(0,jy,kz) ):( ps*Nci+posi(ix+1,jy,kz) );
					int ibefore = (ix==0)?( pos(i-1,j,k)*Nci+posi(Nix-1,jy,kz) ):( ps*Nci+posi(ix-1,jy,kz) );
					int jnext =	  (jy==Niy-1)?( pos(i,j+1,k)*Nci+posi(ix,0,kz) ):( ps*Nci+posi(ix,jy+1,kz) );
					int jbefore = (jy==0)?( pos(i,j-1,k)*Nci+posi(ix,Niy-1,kz) ):( ps*Nci+posi(ix,jy-1,kz) );
					int knext =   (kz==Niz-1)?( pos(i,j,k+1)*Nci+posi(ix,jy,0) ):( ps*Nci+posi(ix,jy,kz+1) );
					int kbefore = (kz==0)?( pos(i,j,k-1)*Nci+posi(ix,jy,Niz-1) ):( ps*Nci+posi(ix,jy,kz-1) );

					double coupleci =  	(CYT[knext].ci-CYT[psi].ci)/(tauit) +
										(CYT[kbefore].ci-CYT[psi].ci)/(tauit) +
										(CYT[jnext].ci-CYT[psi].ci)/(tauit) +
										(CYT[jbefore].ci-CYT[psi].ci)/(tauit) +
										(CYT[inext].ci-CYT[psi].ci)/(CRU2[ps].tauil) +
										(CYT[ibefore].ci-CYT[psi].ci)/(CRU2[pos(i-1,j,k)].tauil);

					double couplecnsr = (CYT[knext].cnsr-CYT[psi].cnsr)/(taunt) +
										(CYT[kbefore].cnsr-CYT[psi].cnsr)/(taunt) +
										(CYT[jnext].cnsr-CYT[psi].cnsr)/(taunt) +
										(CYT[jbefore].cnsr-CYT[psi].cnsr)/(taunt) +
										(CYT[inext].cnsr-CYT[psi].cnsr)/(taunl) +
										(CYT[ibefore].cnsr-CYT[psi].cnsr)/(taunl) ;

					double coupleATP = 	(CBU[knext].caatp-CBU[psi].caatp)/(Datp*tauit) +
										(CBU[kbefore].caatp-CBU[psi].caatp)/(Datp*tauit) +
										(CBU[jnext].caatp-CBU[psi].caatp)/(Datp*tauit) +
										(CBU[jbefore].caatp-CBU[psi].caatp)/(Datp*tauit) +
										(CBU[inext].caatp-CBU[psi].caatp)/(Datp*CRU2[ps].tauil) +
										(CBU[ibefore].caatp-CBU[psi].caatp)/(Datp*CRU2[pos(i-1,j,k)].tauil);

					double coupledye = 	(CBU[knext].cadye-CBU[psi].cadye)/(Ddye*tauit) +
										(CBU[kbefore].cadye-CBU[psi].cadye)/(Ddye*tauit) +
										(CBU[jnext].cadye-CBU[psi].cadye)/(Ddye*tauit) +
										(CBU[jbefore].cadye-CBU[psi].cadye)/(Ddye*tauit) +
										(CBU[inext].cadye-CBU[psi].cadye)/(Ddye*CRU2[ps].tauil) +
										(CBU[ibefore].cadye-CBU[psi].cadye)/(Ddye*CRU2[pos(i-1,j,k)].tauil);

					double coupleegta = (CBU[knext].caegta-CBU[psi].caegta)/(Degta*tauit) +
										(CBU[kbefore].caegta-CBU[psi].caegta)/(Degta*tauit) +
										(CBU[jnext].caegta-CBU[psi].caegta)/(Degta*tauit) +
										(CBU[jbefore].caegta-CBU[psi].caegta)/(Degta*tauit) +
										(CBU[inext].caegta-CBU[psi].caegta)/(Degta*CRU2[ps].tauil) +
										(CBU[ibefore].caegta-CBU[psi].caegta)/(Degta*CRU2[pos(i-1,j,k)].tauil);

					CYT[psi].cinext = CYT[psi].ci 
									  +(	- CYT[psi].Juptake
											- bufftf - buffts - buffcal - buffsr - buffmyo - buffdye - buffegta
											+ coupleci + coupleATP
											+( (crui==0)?( (Vs/Vi)*(diffsi0+diffsiatp0)+(Vp/Vi)*(diffpi0+diffpiatp0) ):0 )
											+( (crui==4)?( (Vs/Vi)*(diffsi1+diffsiatp1)+(Vp/Vi)*(diffpi1+diffpiatp1) ):0 ) 
											+( (crui==1)?( CRU2[ps].Jrelc ):0 )
									  )*DT/( 1.0+katpoff/katpon*Batp/pow2(katpoff/katpon+CYT[psi].ci) );
					
					CYT[psi].cnsrnext = CYT[psi].cnsr 
										+(  CYT[psi].Juptake * Vi/Vnsr
											+ couplecnsr
											+( (crui==0)?( diffjn0*Vjsr/Vnsr ):0 	)
											+( (crui==4)?( diffjn1*Vjsr/Vnsr ):0 	)
											+( (crui==1)?( diffjn1c*Vjsr/Vnsr ):0 )
											+( (crui==5)?( diffjn5c*Vjsr/Vnsr ):0 )
										)*DT;
					
					
					CBU[psi].catf += bufftf*DT;
					CBU[psi].cats += buffts*DT;
					CBU[psi].cacal += buffcal*DT;
					CBU[psi].casr += buffsr*DT;
					CBU[psi].camyo += buffmyo*DT;
					CBU[psi].mgmyo += buffmyomg*DT;
					CBU[psi].caatpnext = Batp*CYT[psi].ci/(katpoff/katpon+CYT[psi].ci);
					CBU[psi].cadyenext += ( buffdye + coupledye
											+( (crui==0)?( (Vs/Vi)*(diffsidye0)+(Vp/Vi)*(diffpidye0) ):0 )
											+( (crui==4)?( (Vs/Vi)*(diffsidye1)+(Vp/Vi)*(diffpidye1) ):0 ) )*DT;
					CBU[psi].caegtanext += ( buffegta + coupleegta
											+( (crui==0)?( (Vs/Vi)*(diffsiegta0)+(Vp/Vi)*(diffpiegta0) ):0 )
											+( (crui==4)?( (Vs/Vi)*(diffsiegta1)+(Vp/Vi)*(diffpiegta1) ):0 ) )*DT;

					if (CYT[psi].cinext < 0 ) 			CYT[psi].cinext = 1e-6;
					if (CYT[psi].cnsrnext < 0 )			CYT[psi].cnsrnext = 1e-6;
					if( CBU[psi].catf < 0 )				CBU[psi].catf = 1e-6;
					if( CBU[psi].cats < 0 )				CBU[psi].cats = 1e-6;
					if( CBU[psi].cacal < 0 )			CBU[psi].cacal = 1e-6;
					if( CBU[psi].casr < 0 )				CBU[psi].casr = 1e-6;
					if( CBU[psi].camyo < 0 )			CBU[psi].camyo = 1e-6;
					if( CBU[psi].mgmyo < 0 )			CBU[psi].mgmyo = 1e-6;
					if( CBU[psi].caatpnext < 0 )		CBU[psi].caatpnext = 1e-6;
					if( CBU[psi].cadyenext < 0 )		CBU[psi].cadyenext = 1e-6;
					if( CBU[psi].caegtanext < 0 )		CBU[psi].caegtanext = 1e-6;
				}
			}
		}
			


		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for( int iii = 0; iii < FINESTEP; ++iii )
		{
			////////////////////// submembrane: dotcs ///////////////////////// 
			double csdiff = ( CRU[pos(i+1,j,k)].cs+CRU[pos(i-1,j,k)].cs-2*CRU[ps].cs )/CRU2[ps].tausl
							+ ( CRU[pos(i,j+1,k)].cs+CRU[pos(i,j-1,k)].cs-2*CRU[ps].cs )/taust
							+ ( CRU[pos(i,j,k+1)].cs+CRU[pos(i,j,k-1)].cs-2*CRU[ps].cs )/taust ;
			double csdiffatp = ( SBU[pos(i+1,j,k)].caatp+SBU[pos(i-1,j,k)].caatp-2*SBU[ps].caatp )/(Datp*CRU2[ps].tausl)
							 + ( SBU[pos(i,j+1,k)].caatp+SBU[pos(i,j-1,k)].caatp-2*SBU[ps].caatp )/(Datp*taust)
							 + ( SBU[pos(i,j,k+1)].caatp+SBU[pos(i,j,k-1)].caatp-2*SBU[ps].caatp )/(Datp*taust);
			double csdiffdye = ( SBU[pos(i+1,j,k)].cadye+SBU[pos(i-1,j,k)].cadye-2*SBU[ps].cadye )/(Ddye*CRU2[ps].tausl)
							 + ( SBU[pos(i,j+1,k)].cadye+SBU[pos(i,j-1,k)].cadye-2*SBU[ps].cadye )/(Ddye*taust)
							 + ( SBU[pos(i,j,k+1)].cadye+SBU[pos(i,j,k-1)].cadye-2*SBU[ps].cadye )/(Ddye*taust);
			double csdiffegta = ( SBU[pos(i+1,j,k)].caegta+SBU[pos(i-1,j,k)].caegta-2*SBU[ps].caegta )/(Degta*CRU2[ps].tausl)
							  + ( SBU[pos(i,j+1,k)].caegta+SBU[pos(i,j-1,k)].caegta-2*SBU[ps].caegta )/(Degta*taust)
							  + ( SBU[pos(i,j,k+1)].caegta+SBU[pos(i,j,k-1)].caegta-2*SBU[ps].caegta )/(Degta*taust);

			double diffps = (CRU2[ps].cpnext-CRU[ps].csnext)/taups;
			double diffpsatp = ( SBU[ps].caatpjnext - SBU[ps].caatpnext )/(taups*Datp);
			double diffpsdye = ( SBU[ps].cadyejnext - SBU[ps].cadyenext )/(taups*Ddye);
			double diffpsegta= ( SBU[ps].caegtajnext - SBU[ps].caegtanext )/(taups*Degta);

			double buffsar = ksaron*CRU[ps].csnext*(Bsar-SBU[ps].casar) - ksaroff*SBU[ps].casar;
			double buffsarh= ksarhon*CRU[ps].csnext*(Bsarh-SBU[ps].casarh) - ksarhoff*SBU[ps].casarh;
			double buffatp = katpon*CRU[ps].csnext*(Batp-SBU[ps].caatpnext) - katpoff*SBU[ps].caatpnext;
			double buffdye = kdyeon*CRU[ps].csnext*(Bdye-SBU[ps].cadyenext) - kdyeoff*SBU[ps].cadyenext;
			double buffegta = kegtaon*CRU[ps].csnext*(Begta-SBU[ps].caegtanext) - kegtaoff*SBU[ps].caegtanext;

			SBU[ps].casar += buffsar*DTF;
			SBU[ps].casarh += buffsarh*DTF;

			SBU[ps].caatpnext += DTF * ( buffatp + diffpsatp*Vp/Vs - diffsiatp0 - diffsiatp1 + csdiffatp );
			SBU[ps].cadyenext += DTF * ( buffdye + diffpsdye*Vp/Vs - diffsidye0 - diffsidye1 + csdiffdye );
			SBU[ps].caegtanext += DTF * ( buffegta + diffpsegta*Vp/Vs - diffsiegta0 - diffsiegta1 + csdiffegta );
			CRU[ps].csnext += DTF*( CRU[ps].JNCX - CRU[ps].Jbg *FracSL* Vp/Vs
									+ diffps*Vp/Vs - diffsi0 - diffsi1 + csdiff
									- buffsar - buffsarh - buffatp - buffdye - buffegta );
			

			////////////////////// proximal space: dotcp ////////////////////// 
			buffsar = ksaron*CRU2[ps].cpnext*(Bsar-SBU[ps].casarj) - ksaroff*SBU[ps].casarj;
			buffsarh = ksarhon*CRU2[ps].cpnext*(Bsarh-SBU[ps].casarhj) - ksarhoff*SBU[ps].casarhj;
			buffatp = katpon*CRU2[ps].cpnext*(Batp-SBU[ps].caatpjnext) - katpoff*SBU[ps].caatpjnext;
			buffdye = kdyeon*CRU2[ps].cpnext*(Bdye-SBU[ps].cadyejnext) - kdyeoff*SBU[ps].cadyejnext;
			buffegta = kegtaon*CRU2[ps].cpnext*(Begta-SBU[ps].caegtajnext) - kegtaoff*SBU[ps].caegtajnext;

			SBU[ps].casarj += buffsar*DTF;
			SBU[ps].casarhj += buffsarh*DTF;

			SBU[ps].caatpjnext += DTF*( buffatp - diffpsatp - diffpiatp0 - diffpiatp1 );
			SBU[ps].cadyejnext += DTF*( buffdye - diffpsdye - diffpidye0 - diffpidye1 );
			SBU[ps].caegtajnext += DTF*( buffegta - diffpsegta - diffpiegta0 - diffpiegta1 );
			CRU2[ps].cpnext += DTF*( CRU2[ps].Jrel + CRU2[ps].Jleak - CRU[ps].JCa - CRU[ps].Jbg*(1.0-FracSL)
									- diffps - diffpi0 - diffpi1 
									- buffsar - buffsarh - buffatp - buffdye - buffegta );
		
			if ( CRU[ps].csnext < 0)			CRU[ps].csnext = 1e-6;
			if ( SBU[ps].casar < 0 )			SBU[ps].casar = 1e-6;
			if ( SBU[ps].casarh < 0 )			SBU[ps].casarh = 1e-6;
			if ( SBU[ps].caatpnext < 0 )		SBU[ps].caatpnext = 1e-6;
			if ( SBU[ps].cadyenext < 0 )		SBU[ps].cadyenext = 1e-6;
			if ( SBU[ps].caegtanext < 0 )		SBU[ps].caegtanext = 1e-6;

			if ( CRU2[ps].cpnext < 0 ) 			CRU2[ps].cpnext = 1e-6;
			if ( SBU[ps].casarj < 0 )			SBU[ps].casarj = 1e-6;
			if ( SBU[ps].casarhj < 0 )			SBU[ps].casarhj = 1e-6;
			if ( SBU[ps].caatpjnext < 0 )		SBU[ps].caatpjnext = 1e-6;
			if ( SBU[ps].cadyejnext < 0 )		SBU[ps].cadyejnext = 1e-6;
			if ( SBU[ps].caegtajnext < 0 )		SBU[ps].caegtajnext = 1e-6;
		}

		// dotcjsr
		for( int iii = 0; iii < FINESTEP; ++iii )
		{
			double betaCSQN = 1.0/( 1.0 + BCSQN*Kc*nCa/pow2(Kc+CRU2[ps].cjsr) );
			CRU2[ps].cjsr += betaCSQN*( -diffjn0-diffjn1 - CRU2[ps].Jrel*Vp/Vjsr - CRU2[ps].Jleak*Vp/Vjsr )*DTF;
			CRU2[ps].Tcj = CRU2[ps].cjsr + BCSQN*nCa*CRU2[ps].cjsr/(Kc+CRU2[ps].cjsr);

			betaCSQN = 1.0/( 1.0 + BCSQN*Kc*nCa/pow2(Kc+CRU2[ps].cjsrc) );
			CRU2[ps].cjsrc += betaCSQN*( -diffjn1c-diffjn5c - CRU2[ps].Jrelc*Vi/Vjsr )*DTF;
			CRU2[ps].Tcjc = CRU2[ps].cjsrc + BCSQN*nCa*CRU2[ps].cjsrc/(Kc+CRU2[ps].cjsrc);

			if (CRU2[ps].cjsrc<0)
				CRU2[ps].Tcjc = 0;
		}

		CRU2[ps].state = localState;
	}
}


__global__ void Finish( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU )
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int ps = pos(i,j,k);
	int ix, jy, kz, psb;

	if((i*j*k)!=0 && i<Nx-1 && j<Ny-1 && k<Nz-1)
	{
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////// update ////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// this update should not be in the function Compute because of synchronization.
		CRU[ps].cs = CRU[ps].csnext;
		SBU[ps].caatp = SBU[ps].caatpnext;
		SBU[ps].cadye = SBU[ps].cadyenext;
		SBU[ps].caegta= SBU[ps].caegtanext;

		CRU2[ps].cp = CRU2[ps].cpnext;
		SBU[ps].caatpj=SBU[ps].caatpjnext;
		SBU[ps].cadyej=SBU[ps].cadyejnext;
		SBU[ps].caegtaj=SBU[ps].caegtajnext;

		for( ix = 0; ix < Nci; ix++ )
		{
			CYT[ps*Nci+ix].ci = CYT[ps*Nci+ix].cinext;
			CYT[ps*Nci+ix].cnsr = CYT[ps*Nci+ix].cnsrnext;
			CBU[ps*Nci+ix].caatp = CBU[ps*Nci+ix].caatpnext;
			CBU[ps*Nci+ix].cadye = CBU[ps*Nci+ix].cadyenext;
			CBU[ps*Nci+ix].caegta = CBU[ps*Nci+ix].caegtanext;
			
		}
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////// Boundary //////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		#ifndef PermeabilizedB

			if (i==1)
			{
				psb=pos(0,j,k);
				for (jy=0;jy<Niy;jy++)
				{
					for (kz=0;kz<Niz;kz++)
					{
						CYT[psb*Nci+posi(Nix-1,jy,kz)].cnsr = CYT[ps*Nci+posi(0,jy,kz)].cnsr;
						CYT[psb*Nci+posi(Nix-1,jy,kz)].ci  =  CYT[ps*Nci+posi(0,jy,kz)].ci;
						CBU[psb*Nci+posi(Nix-1,jy,kz)].caatp= CBU[ps*Nci+posi(0,jy,kz)].caatp;
						CBU[psb*Nci+posi(Nix-1,jy,kz)].cadye= CBU[ps*Nci+posi(0,jy,kz)].cadye;
						CBU[psb*Nci+posi(Nix-1,jy,kz)].caegta=CBU[ps*Nci+posi(0,jy,kz)].caegta;
					}
				}
			}

			if (i==Nx-2)
			{
				psb=pos(Nx-1,j,k);
				for (jy=0;jy<Niy;jy++)
				{
					for (kz=0;kz<Niz;kz++)
					{
						CYT[psb*Nci+posi(0,jy,kz)].cnsr = CYT[ps*Nci+posi(Nix-1,jy,kz)].cnsr;
						CYT[psb*Nci+posi(0,jy,kz)].ci  =  CYT[ps*Nci+posi(Nix-1,jy,kz)].ci;
						CBU[psb*Nci+posi(0,jy,kz)].caatp= CBU[ps*Nci+posi(Nix-1,jy,kz)].caatp;
						CBU[psb*Nci+posi(0,jy,kz)].cadye= CBU[ps*Nci+posi(Nix-1,jy,kz)].cadye;
						CBU[psb*Nci+posi(0,jy,kz)].caegta=CBU[ps*Nci+posi(Nix-1,jy,kz)].caegta;
					}
				}
			}

			if (j==1)
			{
				psb=pos(i,0,k);
				for (ix=0;ix<Nix;ix++)
				{
					for (kz=0;kz<Niz;kz++)
					{
						CYT[psb*Nci+posi(ix,Niy-1,kz)].cnsr = CYT[ps*Nci+posi(ix,0,kz)].cnsr;
						CYT[psb*Nci+posi(ix,Niy-1,kz)].ci  =  CYT[ps*Nci+posi(ix,0,kz)].ci;
						CBU[psb*Nci+posi(ix,Niy-1,kz)].caatp= CBU[ps*Nci+posi(ix,0,kz)].caatp;
						CBU[psb*Nci+posi(ix,Niy-1,kz)].cadye= CBU[ps*Nci+posi(ix,0,kz)].cadye;
						CBU[psb*Nci+posi(ix,Niy-1,kz)].caegta=CBU[ps*Nci+posi(ix,0,kz)].caegta;
					}
				}
				CRU[psb].cs = CRU[ps].cs;
				SBU[psb].caatp = SBU[ps].caatp;
				SBU[psb].cadye = SBU[ps].cadye;
				SBU[psb].caegta= SBU[ps].caegta;
			}

			if (j==Ny-2)
			{
				psb=pos(i,Ny-1,k);
				for (ix=0;ix<Nix;ix++)
				{
					for (kz=0;kz<Niz;kz++)
					{
						CYT[psb*Nci+posi(ix,0,kz)].cnsr = CYT[ps*Nci+posi(ix,Niy-1,kz)].cnsr;
						CYT[psb*Nci+posi(ix,0,kz)].ci  =  CYT[ps*Nci+posi(ix,Niy-1,kz)].ci;
						CBU[psb*Nci+posi(ix,0,kz)].caatp= CBU[ps*Nci+posi(ix,Niy-1,kz)].caatp;
						CBU[psb*Nci+posi(ix,0,kz)].cadye= CBU[ps*Nci+posi(ix,Niy-1,kz)].cadye;
						CBU[psb*Nci+posi(ix,0,kz)].caegta=CBU[ps*Nci+posi(ix,Niy-1,kz)].caegta;
					}
				}
				CRU[psb].cs = CRU[ps].cs;
				SBU[psb].caatp = SBU[ps].caatp;
				SBU[psb].cadye = SBU[ps].cadye;
				SBU[psb].caegta= SBU[ps].caegta;
			}

			if (k==1)
			{
				psb=pos(i,j,0);
				for (ix=0;ix<Nix;ix++)
				{
					for (jy=0;jy<Niy;jy++)
					{
						CYT[psb*Nci+posi(ix,jy,Niz-1)].cnsr = CYT[ps*Nci+posi(ix,jy,0)].cnsr;
						CYT[psb*Nci+posi(ix,jy,Niz-1)].ci  =  CYT[ps*Nci+posi(ix,jy,0)].ci;
						CBU[psb*Nci+posi(ix,jy,Niz-1)].caatp= CBU[ps*Nci+posi(ix,jy,0)].caatp;
						CBU[psb*Nci+posi(ix,jy,Niz-1)].cadye= CBU[ps*Nci+posi(ix,jy,0)].cadye;
						CBU[psb*Nci+posi(ix,jy,Niz-1)].caegta=CBU[ps*Nci+posi(ix,jy,0)].caegta;
					}
				}
				CRU[psb].cs = CRU[ps].cs;
				SBU[psb].caatp = SBU[ps].caatp;
				SBU[psb].cadye = SBU[ps].cadye;
				SBU[psb].caegta= SBU[ps].caegta;
			}

			if (k==Nz-2)
			{
				psb=pos(i,j,Nz-1);
				for (ix=0;ix<Nix;ix++)
				{
					for (jy=0;jy<Niy;jy++)
					{
						CYT[psb*Nci+posi(ix,jy,0)].cnsr = CYT[ps*Nci+posi(ix,jy,Niz-1)].cnsr;
						CYT[psb*Nci+posi(ix,jy,0)].ci  =  CYT[ps*Nci+posi(ix,jy,Niz-1)].ci;
						CBU[psb*Nci+posi(ix,jy,0)].caatp= CBU[ps*Nci+posi(ix,jy,Niz-1)].caatp;
						CBU[psb*Nci+posi(ix,jy,0)].cadye= CBU[ps*Nci+posi(ix,jy,Niz-1)].cadye;
						CBU[psb*Nci+posi(ix,jy,0)].caegta=CBU[ps*Nci+posi(ix,jy,Niz-1)].caegta;
					}
				}
				CRU[psb].cs = CRU[ps].cs;
				SBU[psb].caatp = SBU[ps].caatp;
				SBU[psb].cadye = SBU[ps].cadye;
				SBU[psb].caegta= SBU[ps].caegta;
			}

		#else // Permeabilized cell
			if (i==1)
			{
				psb=pos(0,j,k);
				for (jy=0;jy<Niy;jy++)
				{
					for (kz=0;kz<Niz;kz++)
					{
						CYT[psb*Nci+posi(Nix-1,jy,kz)].cnsr = CYT[ps*Nci+posi(0,jy,kz)].cnsr;
					}
				}
			}

			if (i==Nx-2)
			{
				psb=pos(Nx-1,j,k);
				for (jy=0;jy<Niy;jy++)
				{
					for (kz=0;kz<Niz;kz++)
					{
						CYT[psb*Nci+posi(0,jy,kz)].cnsr = CYT[ps*Nci+posi(Nix-1,jy,kz)].cnsr;
					}
				}
			}

			if (j==1)
			{
				psb=pos(i,0,k);
				for (ix=0;ix<Nix;ix++)
				{
					for (kz=0;kz<Niz;kz++)
					{
						CYT[psb*Nci+posi(ix,Niy-1,kz)].cnsr = CYT[ps*Nci+posi(ix,0,kz)].cnsr;
					}
				}
			}

			if (j==Ny-2)
			{
				psb=pos(i,Ny-1,k);
				for (ix=0;ix<Nix;ix++)
				{
					for (kz=0;kz<Niz;kz++)
					{
						CYT[psb*Nci+posi(ix,0,kz)].cnsr = CYT[ps*Nci+posi(ix,Niy-1,kz)].cnsr;
					}
				}
			}

			if (k==1)
			{
				psb=pos(i,j,0);
				for (ix=0;ix<Nix;ix++)
				{
					for (jy=0;jy<Niy;jy++)
					{
						CYT[psb*Nci+posi(ix,jy,Niz-1)].cnsr = CYT[ps*Nci+posi(ix,jy,0)].cnsr;
					}
				}
			}

			if (k==Nz-2)
			{
				psb=pos(i,j,Nz-1);
				for (ix=0;ix<Nix;ix++)
				{
					for (jy=0;jy<Niy;jy++)
					{
						CYT[psb*Nci+posi(ix,jy,0)].cnsr = CYT[ps*Nci+posi(ix,jy,Niz-1)].cnsr;
					}
				}
			}
		#endif
	}
}

__global__ void	setup_kernel(unsigned long long seed, cru2 *CRU2 )
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	curand_init(seed, pos(i,j,k), 0, &(CRU2[pos(i,j,k)].state)	);
}


__device__ int ryrgating (double cp, double cjsr, curandState *state, int *ncu, int *nou, int *ncb, int *nob, int i, int j, int k, int step,
							double f_Jmax, double alpha, double beta, int Hill, double custar, double cbstar)
{
	curandState localState = *state;

	// should be satisfied: ku*DT < 1.0
	double ku = alpha/( 1.0 + pow(custar/cp,Hill) ) + 2e-6;
	double kb = beta/( 1.0 + pow(cbstar/cp,Hill) ) + 2e-9; // /( 1.0 + pow(500/cjsr,4) )
	double kuminus = 1.0/taucu;
	double kbminus = 1.0/taucb;
	
	double cb = BCSQN * nCa * cjsr / (Kc+cjsr); // bound Ca2+ concentration in JSR
	double ku2b = 1.0/( 1.0+pow(cb/BCSQN/13.3, 24) )/taub;
	double kb2u = 1.0/(  4000.0/( 1.0+pow(cjsr/670.0, 24) ) + 300.0  );
	
	double p_cu_ou = ku * DT;
	double p_cb_ob = kb * DT;
	double p_ou_cu = kuminus * DT;
	double p_ob_cb = kbminus * DT;
	double p_ou_ob = ku2b * DT;
	double p_cu_cb = ku2b * DT;
	double p_cb_cu = kb2u * DT;
	double p_ob_ou = kb2u * (ku/kb) * DT;

	if ( kb < 1e-16 )
	{
		p_ou_ob = 0;
		p_ob_ou = 0;
	}

	int n_cu_ou = number_RyR_transit( &localState, *ncu, p_cu_ou, *ncu );
	int n_cu_cb = number_RyR_transit( &localState, *ncu, p_cu_cb, *ncu - n_cu_ou );
	int n_ou_cu = number_RyR_transit( &localState, *nou, p_ou_cu, *nou );
	int n_ou_ob = number_RyR_transit( &localState, *nou, p_ou_ob, *nou - n_ou_cu );
	int n_cb_cu = number_RyR_transit( &localState, *ncb, p_cb_cu, *ncb );
	int n_cb_ob = number_RyR_transit( &localState, *ncb, p_cb_ob, *ncb - n_cb_cu );
	int n_ob_ou = number_RyR_transit( &localState, *nob, p_ob_ou, *nob );
	int n_ob_cb = number_RyR_transit( &localState, *nob, p_ob_cb, *nob - n_ob_ou );

	*nou += - n_ou_ob - n_ou_cu + n_ob_ou + n_cu_ou;
	*nob += - n_ob_ou - n_ob_cb + n_ou_ob + n_cb_ob;
	*ncu += - n_cu_ou - n_cu_cb + n_ou_cu + n_cb_cu;
	*ncb += - n_cb_cu - n_cb_ob + n_cu_cb + n_ob_cb;

	*state = localState;

	return ( *nou + *nob );
}

// NN: number of RyRs in the current state
// probability: the probability to transit to another state
// upBound: maximum number of RyRs to transit
__device__ int number_RyR_transit(curandState *state, int NN, double probability, int upBound)
{
	int Ntransit = 1001; // larger than nryr
	double mean = NN*probability;

	// If the condition is satisfied, Ntransit is a poisson distribution,
	// otherwise it is a gaussian distribution. They are the approximations 
	// of the binomial distribution.
	if ( probability < 0.26*exp(-NN/2.245) + 0.12*exp(-NN/35.17) + 0.11 )
		while ( Ntransit > upBound ) // Poisson random nmber
		{
			int k = 0;
			double p = 1.0;
			while ( p >= exp(-mean) )
			{
				k++;
				p = p * curand_uniform_double(state);
			}
			Ntransit = k - 1;
		}
	else
		while ( Ntransit < 0 || Ntransit > upBound )
			Ntransit = lrintf( mean + sqrt( mean * (1.0-probability) ) * curand_normal_double(state) );

	return Ntransit;
}


__device__ int LCCgating(double v, double cp, curandState *state, int i )
{	

	curandState localState=*state;

	double dv5 = 5;
	double dvk = 8;

	double fv5 = -22.8;	
	double fvk = 9.1;

	double alphac = 0.22;
	double betac = 4;

	#ifdef ISO
		betac = 2;
		dv5 = 0;
		fv5 = -28;
		fvk = 8.5;
	#endif

	double dinf = 1.0/(1.0+exp(-(v-dv5)/dvk));
	double taud_inverse = 1.0/((1.0-exp(-(v-dv5)/dvk))/(0.035*(v-dv5))*dinf);
	if( (v > -0.0001) && (v < 0.0001) )
		taud_inverse = 0.035*dvk/dinf;
	
	double finf = 1.0-1.0/(1.0+exp(-(v-fv5)/fvk))/(1.+exp((v-60)/12.));
	double tauf_inverse = (0.02-0.007*exp(-pow2(0.0337*(v+10.5))));
	

	double alphad = dinf * taud_inverse;
	double betad = (1.0-dinf) * taud_inverse;
	
	double alphaf = finf * tauf_inverse;
	double betaf = (1.0-finf) * tauf_inverse;
	
	double alphafca = 0.006;
	double betafca = 0.175/( 1 + pow2(60.0/cp) );

	double random = curand_uniform_double(&localState)/DT;
	*state=localState;
	

	if ( i%2 )
		if ( random < alphac )
			return i-1;
		else
			random -= alphac;
	else
		if ( random < betac )
			return i+1;
		else
			random -= betac;
	

	if ( (i/2)%2 )
		if ( random < alphad )
			return i-2;
		else
			random -= alphad;
	else
		if ( random < betad )
			return i+2;
		else
			random -= betad;
	
	
	if ( (i/4)%2 )
		if ( random < alphaf )
			return i-4;
		else
			random -= alphaf;
	else
		if ( random < betaf )
			return i+4;
		else
			random -= betaf;
	
	
	if ( (i/8)%2 )
		if ( random < alphafca )
			return i-8;
		else
			random -= alphafca;
	else
		if ( random < betafca )
			return i+8;
		else
			random -= betafca;

	return (i);
}

__device__ double Single_LCC_Current(double v, double cp) // cp in mM
{
	double ica = 0;
	double za = v*Faraday/RR/Temperature;
	if ( fabs(za)<0.001 ) 
		ica = 2.0*Pca*Faraday*gammai*(cp*exp(2.0*za)-CaO);
	else 
		ica = 4.0*Pca*za*Faraday*gammai*(cp*exp(2.0*za)-CaO)/(exp(2.0*za)-1.0);

	if (ica > 0.0)
		ica = 0.0;

	return ( ica );
}

double Ina( double v, double *hh, double *jj, double *mm, double nai )
{
	double Ena = 1.0/FRT*log(NaO/nai);
	double am = 0.32*(v+47.13)/(1.0-exp(-0.1*(v+47.13)));
	double bm = 0.08*exp(-v/11.0);

	double ah,bh,aj,bj;

	if(v < -40.0)
	{
		ah = 0.135 * exp( -(80.0+v)/6.8 );
		bh = 3.56 * exp(0.079*v) + 310000.0*exp(0.35*v);
		aj = (-127140.0*exp(0.2444*v)-0.00003474*exp(-0.04391*v)) * ( (v+37.78)/(1.0+exp(0.311*(v+79.23))) );
		bj = (0.1212*exp(-0.01052*v))/(1.0+exp(-0.1378*(v+40.14)));
		
	}
	else
	{
		ah = 0.0;
		bh = 1.0/( 0.130*(1.0+exp((v+10.66)/(-11.1))) );
		aj = 0.0;
		bj = ( 0.3*exp(-0.0000002535*v) )/( 1.0 + exp(-0.1*(v+32.0)) );
				
	}
			
	double tauh = 1.0/(ah+bh);
	double tauj = 1.0/(aj+bj);
	double taum = 1.0/(am+bm);

	*hh = ah/(ah+bh)-((ah/(ah+bh))-*hh)*exp(-DT/tauh);
	*jj = aj/(aj+bj)-((aj/(aj+bj))-*jj)*exp(-DT/tauj);
	*mm = am/(am+bm)-((am/(am+bm))-*mm)*exp(-DT/taum);

	double INa = gNa*(*hh)*(*jj)*(*mm)*(*mm)*(*mm)*(v-Ena) + gNaLeak*(v-Ena);

	return INa;
}


double Ikr( double v, double *Xkr )
{
	double krv1 = 0.00138*(v+7.0)/( 1.0-exp(-0.123*(v+7.0))  );
	double krv2 = 0.00061*(v+10.0)/(exp( 0.145*(v+10.0))-1.0);
	double taukr = 1.0/(krv1+krv2);
	double Xkr_inf= 1.0/(1.0+exp(-(v+50.0)/7.5));
	double Rkr = 1.0/(1.0+exp((v+33.0)/22.4));

	*Xkr = Xkr_inf - ( Xkr_inf - *Xkr ) * exp(-DT/taukr);
	
	double I_Kr = gKr * sqrt(KO/5.40) * (*Xkr) * Rkr * (v-Ek);

	return I_Kr;
}

double Iks( double v, double *Xs1, double *Xs2, double *Qks, double cst, double nai )
{
	double prnak = 0.01833;
	double Eks = (1.0/FRT)*log((KO+prnak*NaO)/(KI+prnak*nai));

	double qks_inf =  1.0 + 0.8/( 1.0 + pow((0.5/cst),3) ) ;
	double tauqks = 1000.0;

	double Xs1_inf= 1.0/(1.0+exp(-(v-1.5)/16.7));
	double tauxs = 1.0/( 0.0000719*(v+30.0)/(1.0-exp(-0.148*(v+30.0)))
							+ 0.000131*(v+30.0)/(exp(0.06870*(v+30.0))-1.0) );

	*Xs1 = Xs1_inf-(Xs1_inf-*Xs1)*exp(-DT/tauxs);
	*Xs2 = Xs1_inf-(Xs1_inf-*Xs2)*exp(-DT/tauxs/4.0);
	*Qks = *Qks + DT*( qks_inf-*Qks )/tauqks;

	double I_Ks = gKs*(*Qks)*(*Xs1)*(*Xs2)*(v-Eks);

	return I_Ks;
}

double Ik1( double v )
{
	double Aki = 1.02/(1.0+exp(0.2385*(v-Ek-59.215)));
	double Bki = (0.49124*exp(0.08032*(v-Ek+5.476))+exp(0.061750*(v-Ek-594.31)))/(1.0+exp(-0.5143*(v-Ek+4.753)));
	double I_K1 = gK1 * sqrt(KO/5.4) * Aki/(Aki+Bki) * (v-Ek);

	return I_K1;
}

double Itos(double v, double *Xtos, double *Ytos)
{
	double Xtos_inf = 1.0/( 1.0 + exp( -(v+3.0)/15.0) );
	double Ytos_inf = 1.0/( 1.0 + exp( (v+33.5)/10.0) );
	double Rs_inf = 1.0/( 1.0 + exp( (v+33.5)/10.0) );
	double txs = 9.0/( 1.0 + exp( (v+3.0)/15.0) ) + 0.5;
	double tys = 3000.0/(1.0+exp( (v+60.0)/10.0) ) + 30.0;

	*Xtos = Xtos_inf-(Xtos_inf-*Xtos)*exp(-DT/txs);
	*Ytos = Ytos_inf-(Ytos_inf-*Ytos)*exp(-DT/tys);

	double I_tos = gtos*(*Xtos)*(*Ytos+0.5*Rs_inf)*(v-Ek);

	return I_tos;
}

///////////////// Ito /////////////////

double Itof(double v, double *Xtof, double *Ytof)
{
	double Xtof_inf = 1.0/(1.0+exp( -(v+3.0)/15.0) );
	double Ytof_inf = 1.0/(1.0+exp( (v+33.5)/10.0) );
	double txf = 3.5 * exp( -(v/30.00)*(v/30.0) ) + 1.5;
	double tyf = 20.0/( 1.0+exp( (v+33.5)/10.0 ) )+20.0;

	*Xtof = Xtof_inf-(Xtof_inf-*Xtof)*exp(-DT/txf);
	*Ytof = Ytof_inf-(Ytof_inf-*Ytof)*exp(-DT/tyf);

	double I_tof = gtof*(*Xtof)*(*Ytof)*(v-Ek);

	return I_tof;
}


double Inak( double v, double nai )	 // Mahajan et al 2008
{
	double sigma = ( exp(NaO/67.3) - 1.0 )/7.0;
	double fNaK = 1.0/( 1.0 + 0.1245*exp(-0.1*v*FRT) + 0.0365*sigma*exp(-v*FRT) );
	double I_NaK = gNaK * fNaK * 1.0/( 1.0+pow(12.0/nai,1.0) ) * KO/(KO+1.5);

	return I_NaK;
}	

///////////////////////////	sodium dynamics /////////////////////////////////
double sodium(double v, double nai, double I_Na, double I_NaK, double I_NCX)
{
	// convert pA/pF to mM/ms. Mahajan et al 2008, Eq. 33
	double alpha = 1.0/(2.0*0.096485)/( Vi*Nci*(Nx-2)*(Ny-2)*(Nz-2) ) * Cm / 1000.0;
	double trick = 10.0; // just to speed up Nai dynamics

	double dnai = - trick * alpha * ( I_Na + 3.0*I_NaK + 3.0*I_NCX );

	return (nai + dnai*DT);
}

__device__ double ncx(double v, double cs, double nai, double *Ka)
{
	double csm = cs/1000.0;
	double za = v*Faraday/RR/Temperature;

	double t1 = Kmcai*pow3(NaO)*( 1.0+pow3(nai/Kmnai) );
	double t2 = pow3(Kmnao)*csm*(1.0+csm/Kmcai);
	double t3 = (Kmcao+CaO)*pow3(nai) + csm*pow3(NaO);

	*Ka = 1.0/(1.0+pow3(0.0007/csm));

	double Inaca = Vncx * (*Ka) * ( exp(eta*za)*pow3(nai)*CaO-exp((eta-1.0)*za)*pow3(NaO)*csm )
					/((t1+t2+t3)*(1.0+ksat*exp((eta-1.0)*za)));

	return Inaca;
}	

__device__ double uptake1(double ci, double cnsr)		//uptake
{
	double Ki = 0.2;
	double Knsr = 1700.0;
	double HH = 1.787;
	double Iuptake = Vup * (pow(ci/Ki,HH)-pow(cnsr/Knsr,HH)) / (1.0+pow(ci/Ki,HH)+pow(cnsr/Knsr,HH));
	return Iuptake;
}


__device__ double uptake(double ci, double cnsr)			//uptake
{
	double k1p = 25900/1000.0;
	double k2p = 2540/1000.0;
	double k3p = 20.5/1000.0;
	double k1m = 2*10/1000.0;
	double k2m = 67200*10/1000.0;
	double k3m = 149*10.0/1000.0;
	double kdci = 0.91*4.0;
	double kdcj = 2.24/1.0;//2.24;
	double kdh1 = 1.09e-5;
	double kdhi = 3.54e-3;
	double kdhj = 1.05e-8;
	double kdh = 7.24e-5;

	double MgATP = 4;
	double MgADP = 0.0363;
	double H = 6.3e-8; // pH = 7.2
	double Pii = 10.0;

	double T_MgATP = MgATP/k1m*k1p;
	double T_ci = ci/1000.0/kdci;
	double T_Hi = H/kdhi;
	double T_H1 = H/kdh1;
	double T_cj = cnsr/1000.0/kdcj;
	double T_Hj = H/kdhj;
	double T_H = H/kdh;

	double a1p = k2p * T_MgATP * pow2(T_ci) / ( T_MgATP * pow2(T_ci) + T_Hi*(1+T_MgATP*(1+T_H1+pow2(T_ci))) );
	double a2p = k3p * T_Hj / ( T_Hj * (1+T_H)+ T_H*(1+pow2(T_cj)) );
	double a1m = k2m * MgADP * pow2(T_cj)*T_H / ( T_Hj * (1+T_H) + T_H*(1+pow2(T_cj)) );
	double a2m = k3m * Pii * T_Hi / ( T_MgATP * pow2(T_ci) + T_Hi*( 1+T_MgATP*(1+T_H1 + pow2(T_ci)) ) );

	double Iup = Vup * (a1p*a2p-a1m*a2m) / (a1m+a2m+a1p+a2p);

	return(Iup);
}

#ifdef	output_fluoxt
int convolve(double*fluo, cyt_bu *CBU)
{
	int i,j,k,ix,jy,kz; // i,j,k for CRU index; ix, jy, kz for lattices in each CRU
	double *object,*kernel,*result;
	fftw_complex *hat_object,*hat_kernel,*hat_ko;
	fftw_plan object_forward,kernel_forward,ko_backward;
	
	object =	(double*) fftw_malloc(sizeof(double) * Nx*Nix * Ny*Niy *Nz*Niz);
	kernel =	(double*) fftw_malloc(sizeof(double) * Nx*Nix * Ny*Niy *Nz*Niz);
	result =	(double*) fftw_malloc(sizeof(double) * Nx*Nix * Ny*Niy *Nz*Niz);
	hat_object=	(fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ((Nx*Nix)/2+1) * Ny*Niy * Nz*Niz);
	hat_kernel=	(fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ((Nx*Nix)/2+1) * Ny*Niy * Nz*Niz);
	hat_ko=		(fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ((Nx*Nix)/2+1) * Ny*Niy * Nz*Niz);
	object_forward = 	fftw_plan_dft_r2c_3d(Nz*Niz, Ny*Niy, Nx*Nix, object,	hat_object, FFTW_ESTIMATE);
	kernel_forward = 	fftw_plan_dft_r2c_3d(Nz*Niz, Ny*Niy, Nx*Nix, kernel,	hat_kernel, FFTW_ESTIMATE);
	ko_backward = 		fftw_plan_dft_c2r_3d(Nz*Niz, Ny*Niy, Nx*Nix, hat_ko,	result, FFTW_ESTIMATE);
	
	// initial condition	
	for (k=0; k<Nz; k++) 
	{
		for (kz=0; kz<Niz; kz++)
		{
			for (j=0; j<Ny; j++) 
			{
				for (jy=0; jy<Niy; jy++) 
				{
					for (i=0;i<Nx;i++)
					{
						for (ix=0;ix<Nix;ix++)
						{
							kernel[posall(i*Nix+ix,j*Niy+jy,k*Niz+kz)]=G(i,j,k,ix,jy,kz,Nx*Nix/2,Ny*Niy/2,Nz*Niz/2);
							object[posall(i*Nix+ix,j*Niy+jy,k*Niz+kz)]=CBU[pos(i,j,k)*Nci+posi(ix,jy,kz)].cadye;
						}
					}
				}
			}
		}
	}

	fftw_execute(object_forward);
	fftw_execute(kernel_forward); 
	
	for (k=0; k<Nz*Niz; k++) 
	{
		for (j=0; j<Ny*Niy; j++) 
		{
			for (i=0;i<Nx*Nix/2+1;i++)
			{
				hat_ko[posallf(i,j,k)][0]=hat_object[posallf(i,j,k)][0]*hat_kernel[posallf(i,j,k)][0]
										 -hat_object[posallf(i,j,k)][1]*hat_kernel[posallf(i,j,k)][1];
				hat_ko[posallf(i,j,k)][1]=hat_object[posallf(i,j,k)][0]*hat_kernel[posallf(i,j,k)][1]
										 +hat_object[posallf(i,j,k)][1]*hat_kernel[posallf(i,j,k)][0];
			}
		}
	}
	fftw_execute(ko_backward);

	//Normalization	
	for (k=0; k<Nz*Niz; k++) 
	{
		for (j=0; j<Ny*Niy; j++) 
		{
			for (i=0;i<Nx*Nix;i++)
			{
				result[posall(i,j,k)]=DX*DY*DZ*result[posall(i,j,k)]/((double)(Nx*Nix*Ny*Niy*Nz*Niz)); 
			}
		}
	}
	
	
	// update
	for (k=0; k<Nz*Niz; k++) 
	{
		for (j=0; j<Ny*Niy; j++) 
		{
			for (i=0; i<Nx*Nix; i++)
			{
				int xx = (i+Nx*Nix/2)*(i<Nx*Nix/2) + (i-Nx*Nix/2)*(i>=Nx*Nix/2);
				int yy = (j+Ny*Niy/2)*(j<Ny*Niy/2) + (j-Ny*Niy/2)*(j>=Ny*Niy/2);
				int zz = (k+Nz*Niz/2)*(k<Nz*Niz/2) + (k-Nz*Niz/2)*(k>=Nz*Niz/2);
				fluo[posall(i,j,k)] = result[posall(xx,yy,zz)];
			}
		}
	}
	
	fftw_destroy_plan(object_forward);
	fftw_destroy_plan(kernel_forward);
	fftw_destroy_plan(ko_backward);
	fftw_free(object);
	fftw_free(kernel);
	fftw_free(result);
	fftw_free(hat_object);
	fftw_free(hat_kernel);
	fftw_free(hat_ko);
	return 1;
}
#endif




void matrix2file(cytosol *CYT, int step)
{
	int i,j,k,ix,jy,kz; // i,j,k for CRU index; ix, jy, kz for lattices in each CRU
	double average=0;
	char FileName[50];
	sprintf(FileName,"%-s%d%s","step",step,".vtk");
	
	FILE * file_pointer;
	file_pointer=fopen(FileName,"w");
		fprintf(file_pointer, "# vtk DataFile Version 3.0\n");
		fprintf(file_pointer, "3d\n");
		fprintf(file_pointer, "ASCII\n");
		fprintf(file_pointer, "DATASET STRUCTURED_POINTS\n");
		fprintf(file_pointer, "DIMENSIONS %d %d %d\n",(Nx-2)*Nix,(Ny-2)*Niy,(Nz-2)*Niz);
		fprintf(file_pointer, "ASPECT_RATIO 1 1 1\n");
		fprintf(file_pointer, "ORIGIN 0 0 0\n");
		fprintf(file_pointer, "POINT_DATA %d\n",(Nx-2)*Nix*(Ny-2)*Niy*(Nz-2)*Niz);
		fprintf(file_pointer, "SCALARS ci double 1\n");
		fprintf(file_pointer, "LOOKUP_TABLE default\n\n");
		for (k=1;k<(Nz-1);k++)
		{
			for (kz=0;kz<Niz;kz++)
			{
				for (j=1;j<(Ny-1);j++)
				{
					for (jy=0;jy<Niy;jy++)
					{
						for (i=1;i<(Nx-1);i++)
						{
							for (ix=0;ix<Nix;ix++)
							{
								fprintf(file_pointer,"%g \t", CYT[pos(i,j,k)*Nci+posi(ix,jy,kz)].ci);
								average += CYT[pos(i,j,k)*Nci+posi(ix,jy,kz)].ci;
								if( CYT[pos(i,j,k)*Nci+posi(ix,jy,kz)].ci > 100.0 )
								{
									cout << step*DT <<" "<<i<<" "<<j<<" "<<k<<" "<<posi(ix,jy,kz)<<
									" error! ci="<< CYT[pos(i,j,k)*Nci+posi(ix,jy,kz)].ci << endl;
								}
							}
						}
						fprintf(file_pointer, "\n");
					}
				}
			}
		}
		average /= (1.0*(Nx-2)*(Ny-2)*(Nz-2)*Nix*Niy*Niz);
		printf("Average=%g\t",average);
	fclose(file_pointer);
}
