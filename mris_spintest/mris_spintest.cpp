/**
 * @brief Peforms Spin test and other spatial correlation functions
 *
 *
 * Implementation of geometric transfer matrix (GTM). Also includes Muller-Gartner (MG) and
 * Region-based Voxelwise (RBV) partial volume correction.
 */
/*
 * Original Author: Douglas N. Greve
 *
 * Copyright © 2021 The General Hospital Corporation (Boston, MA) "MGH"
 *
 * Terms and conditions for use, reproduction, distribution and contribution
 * are found in the 'FreeSurfer Software License Agreement' contained
 * in the file 'LICENSE' found in the FreeSurfer distribution, and here:
 *
 * https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
 *
 * Reporting: freesurfer@nmr.mgh.harvard.edu
 *
 */


/*
  BEGINHELP

  ENDHELP
*/

/*
  BEGINUSAGE

  ENDUSAGE
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/utsname.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#include "macros.h"
#include "utils.h"
#include "fio.h"
#include "version.h"
#include "cmdargs.h"
#include "error.h"
#include "diag.h"
#include "mri.h"
#include "mri2.h"
#include "timer.h"
#include "region.h"
#include "resample.h"
#include "mrisurf.h"
#include "mrisutils.h"
#include "randomfields.h"

#ifdef _OPENMP
#include "romp_support.h"
#endif

class MRISspinTest {
public:
  MRIS *sphere=NULL;
  MHT *hash = NULL;
  MRI *ref=NULL, *map=NULL, *refmask=NULL, *mapmask=NULL;
  std::vector<double> SpinCC(float alpha, float beta, float gamma);
  std::vector<std::vector<double>> SpinPerm(int nperm);
  RFS *rfs=NULL;
  unsigned long int seed = -1;
  std::vector<double> PermTest(std::vector<double> cc, std::vector<std::vector<double>> ccperm, int sign);
  int WriteMatrix(char *fname, std::vector<std::vector<double>> m);
  int PrintMatrix(FILE *fp, std::vector<std::vector<double>> m);
  int WriteVector(char *fname, std::vector<double> mrow);
  int PrintVector(FILE *fp, std::vector<double> mrow);
  int PrintVectorGlmFit(FILE *fp, std::vector<double> m);
  int WriteVectorGlmFit(char *fname, std::vector<double> m);
  char *ccpermfile=NULL;
};

static int  parse_commandline(int argc, char **argv);
static void check_options(void);
static void print_usage(void) ;
static void usage_exit(void);
static void print_help(void) ;
static void print_version(void) ;
static void dump_options(FILE *fp);
int main(int argc, char *argv[]) ;

const char *Progname = NULL;
char *cmdline, cwd[2000];
int debug=0;
int checkoptsonly=0;
struct utsname uts;
char *spherefile=NULL;
char *outdir=NULL;
char *reffile = NULL;
int refframe = 0;
char *mapfile = NULL;
char *refmaskfile = NULL;
char *mapmaskfile = NULL;
unsigned long int seed = -1;
int nperm = 0;
int threads = 1;
FILE *logfp=NULL;
char *SUBJECTS_DIR=NULL;
char *ccfile = NULL;
char *glmfitfile = NULL;

/*---------------------------------------------------------------*/
int main(int argc, char *argv[]) 
{
  int nargs,err;
  char logfile[1000];
  char fname[1000];

  nargs = handleVersionOption(argc, argv, "mri_gtmpvc");
  if (nargs && argc - nargs == 1) exit (0);
  argc -= nargs;
  cmdline = argv2cmdline(argc,argv);
  uname(&uts);
  getcwd(cwd,2000);
  SUBJECTS_DIR = getenv("SUBJECTS_DIR");

  Progname = argv[0] ;
  argc --;
  argv++;
  ErrorInit(NULL, NULL, NULL) ;
  DiagInit(NULL, NULL, NULL) ;
  if (argc == 0) usage_exit();
  parse_commandline(argc, argv);
  check_options();
  if (checkoptsonly) return(0);
  dump_options(stdout);

#ifdef _OPENMP
  printf("%d avail.processors, using %d\n",omp_get_num_procs(),omp_get_max_threads());
#endif

  logfp = stdout;
  if(outdir){
    printf("Creating output directory %s\n",outdir);
    err = mkdir(outdir,0777);
    if(err != 0 && errno != EEXIST) {
      printf("ERROR: creating directory %s\n",outdir);
      perror(NULL);
      return(1);
    }
    sprintf(logfile,"%s/mris_spintest.log",outdir);
    logfp = fopen(logfile,"w");
  }
  dump_options(logfp);

  MRISspinTest st;

  st.sphere = MRISread(spherefile);  
  if(!st.sphere) exit(1);
  st.ref = MRIread(reffile);
  if(!st.ref) exit(1);
  st.map = MRIread(mapfile);
  if(!st.map) exit(1);
  if(refmaskfile){
    st.refmask = MRIread(refmaskfile);
    if(!st.refmask) exit(1);
  }
  if(mapmaskfile){
    st.mapmask = MRIread(mapmaskfile);
    if(!st.mapmask) exit(1);
  }
  st.seed = seed;

  // Prune the masks with the data?
  std::vector<double> cc = MRIspatialCC(st.ref,refframe,st.refmask,st.map,st.mapmask);
  if(ccfile) st.WriteVector(ccfile,cc);
  if(glmfitfile) st.WriteVectorGlmFit(glmfitfile,cc);
  if(outdir){
    sprintf(fname,"%s/cc.dat",outdir);
    st.WriteVector(fname,cc);
    sprintf(fname,"%s/cc.glmfit.dat",outdir);
    st.WriteVectorGlmFit(fname,cc);
  }

  Timer timer, mytimer;
  if(nperm > 0){
    sprintf(fname,"%s/cc.perm.dat",outdir);
    st.ccpermfile = fname; // cant change fname here
    std::vector<std::vector<double>> ccperm = st.SpinPerm(nperm);
    for(int permsign = -1; permsign < 2; permsign++){
      std::vector<double> p =  st.PermTest(cc, ccperm, permsign);
      char *permname=NULL;
      if(permsign == -1) permname = (char*)"neg";  
      if(permsign ==  0) permname = (char*)"abs";  
      if(permsign == +1) permname = (char*)"pos";  
      sprintf(fname,"%s/p.%s.dat",outdir,permname);
      st.WriteVector(fname,p);
    }
  }

  fprintf(logfp,"#VMPC# mris_spintest VmPeak  %d\n",GetVmPeak());
  fprintf(logfp,"mris_spintest-runtime %5.2f min\n",timer.minutes());
  fprintf(logfp,"mris_spintest done\n");
  fclose(logfp);
  printf("#VMPC# mris_spintest VmPeak  %d\n",GetVmPeak());
  printf("mris_spintest-runtime %5.2f min\n",timer.minutes());
  printf("mris_spintest done\n");
  return(0);
  exit(0);

} // end of main

/*--------------------------------------------------------------------*/
/*---------------------------------------------------------------*/
/*---------------------------------------------------------------*/
static int parse_commandline(int argc, char **argv) {
  int  nargc , nargsused;
  char **pargv, *option ;

  if (argc < 1) usage_exit();

  nargc   = argc;
  pargv = argv;
  while (nargc > 0) {

    option = pargv[0];
    if(debug) printf("%d %s\n",nargc,option);
    nargc -= 1;
    pargv += 1;

    nargsused = 0;

    if(!strcasecmp(option, "--help"))  print_help() ;
    else if(!strcasecmp(option, "--version")) print_version() ;
    else if(!strcasecmp(option, "--debug"))   debug = 1;
    else if(!strcasecmp(option, "--checkopts"))   checkoptsonly = 1;
    else if(!strcasecmp(option, "--nocheckopts")) checkoptsonly = 0;
    else if(!strcasecmp(option, "--o")) {
      if(nargc < 1) CMDargNErr(option,1);
      outdir = pargv[0];
      nargsused = 1;
    }
    else if(!strcasecmp(option, "--cc")) {
      if(nargc < 1) CMDargNErr(option,1);
      ccfile = pargv[0];
      nargsused = 1;
    }
    else if(!strcasecmp(option, "--cc-glmfit")) {
      if(nargc < 1) CMDargNErr(option,1);
      glmfitfile = pargv[0];
      nargsused = 1;
    }
    else if(!strcasecmp(option, "--sphere")) {
      if(nargc < 1) CMDargNErr(option,1);
      spherefile = pargv[0];
      nargsused = 1;
    }
    else if(!strcasecmp(option, "--ref")) {
      if(nargc < 1) CMDargNErr(option,1);
      reffile = pargv[0];
      nargsused = 1;
    }
    else if(!strcasecmp(option, "--ref-mask")) {
      if(nargc < 1) CMDargNErr(option,1);
      refmaskfile = pargv[0];
      nargsused = 1;
    }
    else if(!strcasecmp(option, "--ref-frame")){
      if(nargc < 1) CMDargNErr(option,1);
      sscanf(pargv[0],"%d",&refframe);
      nargsused = 1;
    }
    else if(!strcasecmp(option, "--map")) {
      if(nargc < 1) CMDargNErr(option,1);
      mapfile = pargv[0];
      nargsused = 1;
    }
    else if(!strcasecmp(option, "--map-mask")) {
      if(nargc < 1) CMDargNErr(option,1);
      mapmaskfile = pargv[0];
      nargsused = 1;
    }
    else if(!strcasecmp(option, "--mask")) {
      if(nargc < 1) CMDargNErr(option,1);
      refmaskfile = pargv[0];
      mapmaskfile = pargv[0];
      nargsused = 1;
    }
    else if(!strcasecmp(option, "--seed")){
      if(nargc < 1) CMDargNErr(option,1);
      sscanf(pargv[0],"%lu",&seed);
      nargsused = 1;
    }
    else if(!strcasecmp(option, "--nperm")){
      if(nargc < 1) CMDargNErr(option,1);
      sscanf(pargv[0],"%d",&nperm);
      nargsused = 1;
    }
    else if(!strcasecmp(option, "--threads")){
      if(nargc < 1) CMDargNErr(option,1);
      sscanf(pargv[0],"%d",&threads);
      #ifdef _OPENMP
      omp_set_num_threads(threads);
      #endif
      nargsused = 1;
    } 
    else if(!strcasecmp(option, "--max-threads")){
      threads = 1;
      #ifdef _OPENMP
      threads = omp_get_max_threads();
      omp_set_num_threads(threads);
      #endif
    } 
    else if(!strcasecmp(option, "--max-threads-1") || !strcasecmp(option, "--max-threads-minus-1")){
      threads = 1;
      #ifdef _OPENMP
      threads = omp_get_max_threads()-1;
      if(threads < 0) threads = 1;
      omp_set_num_threads(threads);
      #endif
    } 
    else {
      fprintf(stderr,"ERROR: Option %s unknown\n",option);
      if(CMDsingleDash(option))
        fprintf(stderr,"       Did you really mean -%s ?\n",option);
      exit(-1);
    }
    nargc -= nargsused;
    pargv += nargsused;
  }
  return(0);
}
/*---------------------------------------------------------------*/
static void usage_exit(void) {
  print_usage() ;
  exit(1) ;
}
/*---------------------------------------------------------------*/
static void print_usage(void) {
  printf("USAGE: %s \n",Progname) ;
  printf("\n");
  printf("   --o outdir\n");
  printf("   --sphere spheresurf\n");
  printf("   --ref refmap\n");
  printf("   --map maps\n");
  printf("   --ref-mask refmask\n");
  printf("   --map-mask mapmask\n");
  printf("   --seed seed\n");
  printf("   --ref-frame frameno (0-based)\n");
  printf("   --nperm nperm\n");
  printf("   --cc ccfile\n");
  printf("   --cc-glmfit ccglmfitfile\n");
  // revmapflag, dojac, --s subject hemi
  #ifdef _OPENMP
  printf("   --threads N : use N threads (with Open MP)\n");
  printf("   --max-threads : use the maximum allowable number of threads for this computer\n");
  printf("   --max-threads-minus-1 : use one less than the maximum allowable number of threads for this computer\n");
  #endif
  printf("   --sd SUBJECTS_DIR\n");
  printf("   --gdiag diagno : set diagnostic level\n");
  printf("   --debug     turn on debugging\n");
  printf("   --checkopts don't run anything, just check options and exit\n");
  printf("   --help      print out information on how to use this program\n");
  printf("   --version   print out version and exit\n");
  printf("\n");
  std::cout << getVersion() << std::endl;
  printf("\n");
}
/*---------------------------------------------------------------*/
static void print_help(void) {
  print_usage() ;
  exit(1) ;
}
/*---------------------------------------------------------------*/
static void print_version(void) {
  std::cout << getVersion() << std::endl;
  exit(1) ;
}
/*---------------------------------------------------------------*/
static void check_options(void) 
{
  if(nperm > 0){
    if(outdir == NULL){
      printf("ERROR: must spec outdir\n");
      exit(1);
    }
  }
  else {
    if(ccfile == NULL){
      printf("ERROR: must spec an outdir or a ccfile\n");
      exit(1);
    }
  }
  if(spherefile == NULL){
    printf("ERROR: must spec sphere\n");
    exit(1);
  }
  if(reffile == NULL){
    printf("ERROR: must spec ref\n");
    exit(1);
  }
  if(mapfile == NULL){
    printf("ERROR: must spec ref\n");
    exit(1);
  }
  return;
}
/*---------------------------------------------------------------*/
static void dump_options(FILE *fp) {
  fprintf(fp,"\n");
  fprintf(fp,"%s\n", getVersion().c_str());
  fprintf(fp,"setenv SUBJECTS_DIR %s\n",SUBJECTS_DIR);
  fprintf(fp,"cd %s\n",cwd);
  fprintf(fp,"%s\n",cmdline);
  fprintf(fp,"sysname  %s\n",uts.sysname);
  fprintf(fp,"hostname %s\n",uts.nodename);
  fprintf(fp,"machine  %s\n",uts.machine);
  fprintf(fp,"user     %s\n",VERuser());
  fprintf(fp,"ref   %s\n",reffile);
  if(refmaskfile) fprintf(fp,"refmask  %s\n",refmaskfile);
  fprintf(fp,"refframe %d\n",refframe);
  fprintf(fp,"map   %s\n",mapfile);
  if(mapmaskfile) fprintf(fp,"mapmask  %s\n",mapmaskfile);
  fprintf(fp,"nperm   %d\n",nperm);
  fprintf(fp,"seed   %lu\n",seed);
  return;
}


std::vector<std::vector<double>> MRISspinTest::SpinPerm(int nperm)
{
  std::vector<std::vector<double>> ccperm(nperm,std::vector<double>(3+map->nframes));
  std::vector<double> alphalist(nperm), betalist(nperm), gammalist(nperm);

  if(!rfs){
    // Initialize the random number generator
    rfs = RFspecInit(0, NULL);
    rfs->name = strcpyalloc("uniform");
    rfs->params[0] = 0;
    rfs->params[1] = 1;
    RFspecSetSeed(rfs, seed);
  }

  // Create the list of random angles serially/deterministically
  for(int n=0; n < nperm; n++){
    alphalist[n] = 360*RFdrawVal(rfs);
    betalist[n]  = 360*RFdrawVal(rfs);
    gammalist[n] = 360*RFdrawVal(rfs);
    // allocate vects for cperm
    std::vector<double> cc(3+map->nframes);
    ccperm[n] = cc;
  }

  // Note: when printing out ccperm within the loop, you get to see
  // what is happening as it happens, but, while the results are
  // deterministic, the order may be different if nthreads > 1.
  FILE *fp=NULL;
  if(ccpermfile) fp = fopen(ccpermfile,"w");

  // ccperm will have 3+nframes columns. The first 3 will be the
  // rotations, then the nframes will be the spatial correlations.
  // Keeping the rotations allows computing stats on the
  // randomizations.
  #ifdef HAVE_OPENMP
  #pragma omp parallel for 
  #endif
  for(int n=0; n < nperm; n++){
    double alpha = alphalist[n];
    double beta = betalist[n];
    double gamma = gammalist[n];
    std::vector<double> cc = SpinCC(alpha,beta,gamma);
    ccperm[n][0] = alpha;
    ccperm[n][1] = beta;
    ccperm[n][2] = gamma;
    for(int k=0; k < cc.size(); k++) ccperm[n][3+k] = cc[k];
    printf("perm n=%d vmp=%d  ",n,GetVmPeak()); PrintVector(stdout,ccperm[n]);
    if(fp) PrintVector(fp,ccperm[n]);
  }
  if(fp) fclose(fp);

  return(ccperm);
}

std::vector<double> MRISspinTest::PermTest(std::vector<double> cc, std::vector<std::vector<double>> ccperm, int sign)
{
  printf("PermTest\n");
  std::vector<double> p(cc.size());
  std::vector<int>  nhits(cc.size());
  for(int n=0; n < ccperm.size(); n++){
    for(int k=0; k < cc.size();k++){
      int hit = 0;
      double ccpk = ccperm[n][k+3];
      if(sign >  0 && ccpk > cc[k]) hit=1;
      if(sign == 0 && fabs(ccpk) > fabs(cc[k])) hit=1;
      if(sign <  0 && ccpk < cc[k]) hit=1;
      nhits[k] += hit;
      if(hit) printf("n=%d k=%d %d %g %g\n",n,k,nhits[k],ccpk,cc[k]);
    }
  }
  for(int k=0; k < cc.size();k++) {
    p[k] = (double)nhits[k]/ccperm.size();
    printf("k=%d p=%g\n",k,p[k]);
  }
  return(p);
}

std::vector<double>  MRISspinTest::SpinCC(float alphadeg, float betadeg, float gammadeg)
{
  // This could probably be sped up condiserably
  // alpha is rotation about z
  // beta is rotation about y
  // gamma is rotation about x
  // Note that these are rotations, not spherical coords
  double alpha = alphadeg*M_PI/180;
  double beta  = betadeg*M_PI/180;
  double gamma = gammadeg*M_PI/180;
  MRIS *sphererot = MRISrotate(sphere, NULL, alpha, beta, gamma);
  MRIS *surfs[2] = {sphere,sphererot};
  int ReverseMapFlag = 1;
  int DoJac = 0;
  int UseHash = 1;
  int refframe = 0;
  //MHT **hashvect;
  //hashvect = (MHT **)calloc(sizeof(MHT *), 1);
  //hashvect[0] = hash;
  MHT_maybeParallel_begin();
  MRI *refrot = MRISapplyReg(ref, surfs , 2, ReverseMapFlag, DoJac, UseHash);
  MRI *refrotmask = NULL;
  if(refmask) refrotmask = MRISapplyReg(refmask, surfs , 2, 0, 0, UseHash);
  MHT_maybeParallel_end();
  std::vector<double> cc = MRIspatialCC(refrot,refframe,refrotmask,map,mapmask);
  MRIfree(&refrot);
  if(refrotmask) MRIfree(&refrotmask);
  MRISfree(&sphererot);
  return(cc);
}

int MRISspinTest::WriteMatrix(char *fname, std::vector<std::vector<double>> m)
{
  FILE *fp = fopen(fname,"w");
  PrintMatrix(fp,m);
  fclose(fp);
  return(0);
}
int MRISspinTest::WriteVector(char *fname, std::vector<double> mrow)
{
  FILE *fp = fopen(fname,"w");
  PrintVector(fp,mrow);
  fclose(fp);
  return(0);
}
int MRISspinTest::PrintMatrix(FILE *fp, std::vector<std::vector<double>> m)
{
  for(int n=0; n < m.size(); n++){
    std::vector<double> mrow = m[n];
    PrintVector(fp,mrow);
  }
  return(0);
}
int MRISspinTest::PrintVector(FILE *fp, std::vector<double> mrow)
{
  for(int k=0; k < mrow.size(); k++) fprintf(fp,"%8.4f ",mrow[k]);
  fprintf(fp,"\n"); 
  fflush(fp);
  return(0);
}
int MRISspinTest::PrintVectorGlmFit(FILE *fp, std::vector<double> m)
{
  fprintf(fp,"RowNo Val\n");
  for(int c=0; c < m.size(); c++) fprintf(fp,"%2d %8.4lf\n",c,m[c]);
  return(0);
}
int MRISspinTest::WriteVectorGlmFit(char *fname, std::vector<double> m)
{
  FILE *fp = fopen(fname,"w");
  this->PrintVectorGlmFit(fp,m);
  return(0);
}
