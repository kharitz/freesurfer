/**
 * @brief Creates a segmentation used with the geometric transfer matrix.
 *
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

#undef X
#include "mri2.h"
#include "macros.h"
#include "utils.h"
#include "fio.h"
#include "version.h"
#include "cmdargs.h"
#include "error.h"
#include "diag.h"
#include "timer.h"
#include "mri_identify.h"

#include "romp_support.h"

#include "gtm.h"
#include "resample.h"

static int  parse_commandline(int argc, char **argv);
static void check_options(void);
static void print_usage(void) ;
static void usage_exit(void);
static void print_help(void) ;
static void print_version(void) ;
static void dump_options(FILE *fp);
int main(int argc, char *argv[]) ;
MRI *MRIErodeWMSeg(MRI *seg, int nErode3d, MRI *outseg);

const char *Progname = NULL;
char *cmdline, cwd[2000];
int debug=0;
int checkoptsonly=0;
struct utsname uts;

GTMSEG *gtmseg;
char *OutVolFile=NULL;
char *SUBJECTS_DIR;
int nthreads;
COLOR_TABLE *ctMaster, *ctMerge=NULL;
int nErodeWM = 0, ErodeWMTopo=1;

/*---------------------------------------------------------------*/
int main(int argc, char *argv[]) {
  int nargs,err;
  char tmpstr[5000],*stem;
  Timer mytimer;
  COLOR_TABLE *ct;
  MRI *mritmp;
  LTA *seg2new, *ltatmp;

  nargs = handleVersionOption(argc, argv, "mri_gtmseg");
  if (nargs && argc - nargs == 1) exit (0);
  argc -= nargs;
  cmdline = argv2cmdline(argc,argv);
  uname(&uts);
  getcwd(cwd,2000);
  SUBJECTS_DIR = getenv("SUBJECTS_DIR");

  gtmseg = (GTMSEG *) calloc(sizeof(GTMSEG),1);
  gtmseg->USF = 2;
  gtmseg->OutputUSF = gtmseg->USF;
  gtmseg->dmax = 5.0;
  gtmseg->KeepHypo = 0;
  gtmseg->KeepCC = 0;
  gtmseg->apasfile = "apas+head.mgz";
  gtmseg->ctxannotfile = "aparc.annot";
  gtmseg->ctxlhbase = 1000;
  gtmseg->ctxrhbase = 2000;
  gtmseg->SubSegWM = 0;
  if(gtmseg->SubSegWM){
    gtmseg->wmannotfile = "lobes.annot";
    gtmseg->wmlhbase =  3200;
    gtmseg->wmrhbase =  4200;
  }
  else  gtmseg->wmannotfile = NULL;
  gtmseg->nlist = 0;
  gtmseg->lhmin = 1000;
  gtmseg->lhmax = 1900;
  gtmseg->rhmin = 2000;
  gtmseg->rhmax = 2900;

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

  ctMaster = TissueTypeSchema(NULL,"default-apr-2019+head");

#if 0 // this stuff is better handled by a merge ct right now as TT is not embedded
  if(0 && gtmseg->mergesegfile){
    sprintf(tmpstr, "%s/%s/mri/%s", SUBJECTS_DIR, gtmseg->subject, gtmseg->mergesegfile);
    printf("Loading header %s\n", tmpstr);
    MRI *mergeseg = MRIreadHeader(tmpstr,MRI_VOLUME_TYPE_UNKNOWN);
    if(mergeseg == NULL) return (1);
    if(mergeseg->ct){
      printf("Merging ctab %d %d\n",ctMaster->nentries,mergeseg->ct->nentries);
      for(int n=0; n < gtmseg->mergesegids.size(); n++){
	// may need to expand the number of segs in the ctab if largest merge seg larger
	int segid = gtmseg->mergesegids[n];
	COLOR_TABLE_ENTRY *cte2 = mergeseg->ct->entries[segid];
	if(!cte2) {
	  printf("ERROR: embedded ctab in mergeseg does not have %d\n",gtmseg->mergesegids[n]);
	  return(1);
	}
	COLOR_TABLE_ENTRY *cte1 = ctMaster->entries[segid];
	if(!cte1) {
	  cte1 = (CTE*) calloc(sizeof(CTE),1);
	  ctMaster->entries[segid] = cte1;
	}
	memcpy(cte1, cte2, sizeof(CTE));
	// Currently, tissue type does not work because it cannot be
	// written into or read from embedded ctab. Use merge ctab instead.
	//if(mergeseg->ct->ctabTissueType == 0) cte1->TissueType = -1;
	printf("%d %s  %d\n",segid,cte1->name,cte1->TissueType);
      }
    }
    MRIfree(&mergeseg);
    //CTABwriteFileASCIItt(ctMaster,"dng.ctab");
  }
#endif

  if(ctMerge) {
    printf("Merging CTAB master with merge ctab\n");
    CTABmerge(ctMaster,ctMerge);
  }
  printf("master tissue type schema %s\n",ctMaster->TissueTypeSchema);

  mytimer.reset();
  printf("Starting MRIgtmSeg()\n"); fflush(stdout);
  err = MRIgtmSeg(gtmseg);
  if(err) exit(1);

  if(nErodeWM > 0){
    printf("Eroding WM %d iterations, topo=%d\n",nErodeWM,ErodeWMTopo); fflush(stdout);
    MRIerodeAndReplaceLabel(gtmseg->seg,  2, 5001, ErodeWMTopo, nErodeWM, gtmseg->seg);
    MRIerodeAndReplaceLabel(gtmseg->seg, 41, 5002, ErodeWMTopo, nErodeWM, gtmseg->seg);
  }

  printf("Computing colortable\n");
  ct = GTMSEGctab(gtmseg, ctMaster);
  if(ct == NULL) exit(1);
  //CTABwriteFileASCIItt(ct,"dng.ctab");
  //printf("wrote dng.ctab\n\n\n");
  printf("ct version %d\n",ct->version);

  if(gtmseg->OutputUSF != gtmseg->USF){
    printf("Changing size of output to USF of %d\n",gtmseg->OutputUSF);
    double res = 1.0/gtmseg->OutputUSF;
    mritmp = MRIchangeSegRes(gtmseg->seg, res,res,res, ct, &seg2new);
    if(mritmp == NULL) exit(1);
    MRIfree(&gtmseg->seg);
    gtmseg->seg = mritmp;
    ltatmp = LTAconcat2(gtmseg->anat2seg,seg2new,1);
    LTAfree(&gtmseg->anat2seg);
    LTAfree(&seg2new);
    gtmseg->anat2seg = ltatmp;
    printf("Recomputing colortable\n");
    CTABfree(&ct);
    ct = GTMSEGctab(gtmseg, ctMaster);
    if(ct == NULL) exit(1);
  }
  printf("tissue type schema %s\n",ct->TissueTypeSchema);

  if(nErodeWM > 0){
    int structure; COLOR_TABLE_ENTRY *cte;
    structure = 5001; cte = ct->entries[structure]; sprintf(cte->name,"Left-Shell-Cerebral-White-Matter");
    structure = 5002; cte = ct->entries[structure]; sprintf(cte->name,"Right-Shell-Cerebral-White-Matter");
  }

  // embed color table in segmentation
  gtmseg->seg->ct = CTABdeepCopy(ct);

  sprintf(tmpstr,"%s/%s/mri/%s",SUBJECTS_DIR,gtmseg->subject,OutVolFile);
  printf("Writing output file to %s\n",tmpstr);
  err = MRIwrite(gtmseg->seg,tmpstr);
  if(err) exit(1);
  stem = IDstemFromName(OutVolFile);

  sprintf(tmpstr,"%s/%s/mri/%s.ctab",SUBJECTS_DIR,gtmseg->subject,stem);
  printf("Writing colortable to %s\n",tmpstr);
  err = CTABwriteFileASCIItt(ct,tmpstr);
  if(err) exit(1);

  sprintf(tmpstr,"%s/%s/mri/%s.lta",SUBJECTS_DIR,gtmseg->subject,stem);
  printf("Writing lta file to %s\n",tmpstr);
  err=LTAwrite(gtmseg->anat2seg,tmpstr);
  if(err) exit(1);



  printf("mri_gtmseg finished in %g minutes\n", mytimer.minutes());
  printf("mri_gtmseg done\n");fflush(stdout);

  exit(0);  return 0;
}
/*------------------------------------------------------------------*/
/*------------------------------------------------------------------*/
/*------------------------------------------------------------------*/

static int parse_commandline(int argc, char **argv) {
  int  nargc , nargsused;
  char **pargv, *option ;

  if (argc < 1) usage_exit();

  nargc   = argc;
  pargv = argv;
  while (nargc > 0) {

    option = pargv[0];
    if (debug) printf("%d %s\n",nargc,option);
    nargc -= 1;
    pargv += 1;

    nargsused = 0;

    if (!strcasecmp(option, "--help"))  print_help() ;
    else if (!strcasecmp(option, "--version")) print_version() ;
    else if (!strcasecmp(option, "--debug"))   debug = 1;
    else if (!strcasecmp(option, "--checkopts"))   checkoptsonly = 1;
    else if (!strcasecmp(option, "--nocheckopts")) checkoptsonly = 0;

    else if (!strcasecmp(option, "--o")) {
      if (nargc < 1) CMDargNErr(option,1);
      OutVolFile = pargv[0];
      nargsused = 1;
    } 

    else if(!strcasecmp(option, "--s")) {
      if(nargc < 1) CMDargNErr(option,1);
      gtmseg->subject = pargv[0];
      nargsused = 1;
    } 
    else if(!strcasecmp(option, "--usf") || !strcasecmp(option, "--internal-usf")) {
      if(nargc < 1) CMDargNErr(option,1);
      sscanf(pargv[0],"%d",&gtmseg->USF);
      nargsused = 1;
    } 
    else if(!strcasecmp(option, "--wm-erode")){
      if(nargc < 1) CMDargNErr(option,1);
      sscanf(pargv[0],"%d",&nErodeWM);
      nargsused = 1;
    } 
    else if(!strcasecmp(option, "--wm-erode-topo")){
      if(nargc < 1) CMDargNErr(option,1);
      sscanf(pargv[0],"%d",&ErodeWMTopo);
      nargsused = 1;
    } 
    else if(!strcasecmp(option, "--output-usf")) {
      if(nargc < 1) CMDargNErr(option,1);
      sscanf(pargv[0],"%d",&gtmseg->OutputUSF);
      nargsused = 1;
    } 
    else if (!strcasecmp(option, "--apas")) {
      if (nargc < 1) CMDargNErr(option,1);
      gtmseg->apasfile = pargv[0];
      nargsused = 1;
    } 
    else if (!strcasecmp(option, "--ctab")) {
      if (nargc < 1) CMDargNErr(option,1);
      ctMerge = CTABreadASCII(pargv[0]);
      if(ctMerge == NULL) exit(1);
      nargsused = 1;
    } 
    else if(!strcasecmp(option, "--ctx-annot")) {
      if(nargc < 3) CMDargNErr(option,3);
      gtmseg->ctxannotfile = pargv[0];
      sscanf(pargv[1],"%d",&gtmseg->ctxlhbase);
      sscanf(pargv[2],"%d",&gtmseg->ctxrhbase);
      nargsused = 3;
    } 

    else if(!strcasecmp(option, "--subseg-wm"))    {
      gtmseg->SubSegWM = 1;
      gtmseg->wmannotfile = "lobes.annot";
      gtmseg->wmlhbase =  3200;
      gtmseg->wmrhbase =  4200;
    }
    else if(!strcasecmp(option, "--no-subseg-wm")) {
      gtmseg->wmannotfile = NULL;
      gtmseg->SubSegWM = 0;
    }
    else if(!strcasecmp(option, "--wm-annot")) {
      if(nargc < 3) CMDargNErr(option,3);
      gtmseg->wmannotfile = pargv[0];
      sscanf(pargv[1],"%d",&gtmseg->wmlhbase);
      sscanf(pargv[2],"%d",&gtmseg->wmrhbase);
      gtmseg->SubSegWM = 1;
      nargsused = 3;
    } 
    else if(!strcasecmp(option, "--dmax")) {
      if(nargc < 1) CMDargNErr(option,1);
      sscanf(pargv[0],"%f",&gtmseg->dmax);
      nargsused = 1;
    } 
    else if(!strcasecmp(option, "--keep-hypo"))    gtmseg->KeepHypo = 1;
    else if(!strcasecmp(option, "--no-keep-hypo")) gtmseg->KeepHypo = 0;
    else if(!strcasecmp(option, "--keep-cc"))    gtmseg->KeepCC = 1;
    else if(!strcasecmp(option, "--no-keep-cc")) gtmseg->KeepCC = 0;

    else if (!strcmp(option, "--sd") || !strcmp(option, "-SDIR")) {
      if(nargc < 1) CMDargNErr(option,1);
      setenv("SUBJECTS_DIR",pargv[0],1);
      SUBJECTS_DIR = getenv("SUBJECTS_DIR");
      nargsused = 1;
    } 

    else if (!strcasecmp(option, "--threads")){
      if(nargc < 1) CMDargNErr(option,1);
      sscanf(pargv[0],"%d",&nthreads);
      #ifdef HAVE_OPENMP
      omp_set_num_threads(nthreads);
      #endif
      nargsused = 1;
    } 
    else if (!strcasecmp(option, "--max-threads")){
      nthreads = 1;
      #ifdef HAVE_OPENMP
      nthreads = omp_get_max_threads();
      omp_set_num_threads(nthreads);
      #endif
    } 
    else if (!strcasecmp(option, "--max-threads-1")){
      nthreads = 1;
      #ifdef HAVE_OPENMP
      nthreads = omp_get_max_threads()-1;
      if(nthreads < 0) nthreads = 1;
      omp_set_num_threads(nthreads);
      #endif
    } 
    else if(!strcasecmp(option, "--lhminmax")) {
      if(nargc < 2) CMDargNErr(option,2);
      sscanf(pargv[0],"%d",&gtmseg->lhmin);
      sscanf(pargv[1],"%d",&gtmseg->lhmax);
      nargsused = 2;
    } 
    else if(!strcasecmp(option, "--rhminmax")) {
      if(nargc < 2) CMDargNErr(option,2);
      sscanf(pargv[0],"%d",&gtmseg->rhmin);
      sscanf(pargv[1],"%d",&gtmseg->rhmax);
      nargsused = 2;
    } 
    else if(!strcasecmp(option, "--merge")) {
      if(nargc < 2) CMDargNErr(option,2);
      gtmseg->mergesegfile = pargv[0];
      int nth = 1;
      while(CMDnthIsArg(nargc, pargv, nth) ){
	int segid;
	sscanf(pargv[nth],"%d",&segid);
	gtmseg->mergesegids.push_back(segid);
	nth ++;
      }
      nargsused = nth;
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
/*-----------------------------------------------------------------------------------*/
static void usage_exit(void) {
  print_usage() ;
  exit(1) ;
}
/*-----------------------------------------------------------------------------------*/
static void print_usage(void) {
  printf("USAGE: %s \n",Progname) ;
  printf("\n");
  printf("   --o outvol  : output volume (output will be subject/mri/outvol\n");
  printf("   --s subject : source subject \n");
  printf("   --internal-usf USF : upsampling factor (default %d)\n",gtmseg->USF);
  printf("   --apas apasfile : defines extra-cerebral and subcortical segmentations (%s)\n",gtmseg->apasfile);
  printf("   --ctx-annot annot lhbase rhbase : use annot to segment cortex (%s,%d,%d)\n",
	 gtmseg->ctxannotfile,gtmseg->ctxlhbase,gtmseg->ctxrhbase);
  printf("   --subseg-wm : turn on segmenting of WM into smaller parts (off by default)\n");
  printf("       sets wmannot to lobes.annot and lhbase,rhbase to 3200,4200\n");
  printf("   --wm-annot annot lhbase rhbase : use annot to subseg wm\n");
  printf("   --dmax dmax : distance from ctx for wmseg to be considered 'unsegmented' (%f)\n",gtmseg->dmax);
  printf("   --keep-hypo : do not convert WM hypointensities to a white matter label \n");
  printf("   --keep-cc : do not convert corpus callosum to a white matter label \n");
  printf("   --wm-erode N : erode WM (2,41) by N, replacing with 5001 and 5002\n");
  printf("   --wm-erode-topo topo : topology = 1,2,3 when eroding WM (default is %d)\n",ErodeWMTopo);
  printf("   --ctab ctab.lut : copy items in ctab.lut into master ctab merging or overwriting what is there \n");
  printf("   --lhminmax lhmin lhmax : for defining ribbon in apas (default: %d %d) \n",gtmseg->lhmin,gtmseg->lhmax);
  printf("   --rhminmax rhmin rhmax : for defining ribbon in apas (default: %d %d) \n",gtmseg->rhmin,gtmseg->rhmax);
  printf("   --output-usf OutputUSF : set actual output resolution. Default is to be the same as the --internal-usf");
  printf("   --merge segname segid1 ... : mri/segname");
  printf("\n");
  #ifdef HAVE_OPENMP
  printf("   --threads N : use N threads (with Open MP)\n");
  printf("   --threads-max : use the maximum allowable number of threads for this computer\n");
  printf("   --threads-max-1 : use one less than the maximum allowable number of threads for this computer\n");
  #endif
  printf("   --debug     turn on debugging\n");
  printf("   --checkopts don't run anything, just check options and exit\n");
  printf("   --help      print out information on how to use this program\n");
  printf("   --version   print out version and exit\n");
  printf("\n");
  std::cout << getVersion() << std::endl;
  printf("\n");
}
/*-----------------------------------------------------------------------------------*/
static void print_help(void) {
  print_usage() ;
  printf("This program creates a segmentation that can be used with the geometric transfer matrix (GTM).\n");
  exit(1) ;
}
/*-----------------------------------------------------------------------------------*/
static void print_version(void) {
  std::cout << getVersion() << std::endl;
  exit(1) ;
}
/*-----------------------------------------------------------------------------------*/
static void check_options(void) {
  return;
}
/*-----------------------------------------------------------------------------------*/
static void dump_options(FILE *fp) {
  fprintf(fp,"\n");
  fprintf(fp,"%s\n", getVersion().c_str());
  fprintf(fp,"cwd %s\n",cwd);
  fprintf(fp,"cmdline %s\n",cmdline);
  fprintf(fp,"sysname  %s\n",uts.sysname);
  fprintf(fp,"hostname %s\n",uts.nodename);
  fprintf(fp,"machine  %s\n",uts.machine);
  fprintf(fp,"user     %s\n",VERuser());
  GTMSEGprint(gtmseg, stdout);
  return;
}

/*!
  This is probably not needed anymore as I did not know that it was
  here and re-implemented it as MRIerodeAndReplaceLabel()
  \fn MRI *MRIErodeWMSeg(MRI *seg, int nErode3d, MRI *outseg)
  Takes a segmentation and erodes WM by nErode3d, then sets the
  seg value in surviving voxels to 5001 or 5002, which is
  UnsegmentedWhiteMatter in the LUT. Other WM voxels are not
  changed. It assumes that WM is 2 and 41.
 */
MRI *MRIErodeWMSeg(MRI *seg, int nErode3d, MRI *outseg)
{
  int c,n;
  MRI *wm;

  if(outseg == NULL){
    outseg  = MRIallocSequence(seg->width, seg->height, seg->depth, MRI_INT, 1);
    MRIcopyHeader(seg,outseg);
    MRIcopyPulseParameters(seg,outseg);
  }
  MRIcopy(seg,outseg);

  wm = MRIallocSequence(seg->width, seg->height, seg->depth, MRI_INT, 1);
#ifdef HAVE_OPENMP
  #pragma omp parallel for 
#endif
  for(c=0; c < seg->width; c++) {
    int r,s;
    int val;
    for(r=0; r < seg->height; r++) {
      for(s=0; s < seg->depth; s++) {
	val = MRIgetVoxVal(seg,c,r,s,0);
	if(val != 2 && val != 41) continue;
	MRIsetVoxVal(wm,c,r,s,0,1);
      }
    }
  }
  MRIwrite(wm,"wm0.mgh");

  for(n=0; n<nErode3d; n++) MRIerode(wm,wm);
  MRIwrite(wm,"wm.erode.mgh");

#ifdef HAVE_OPENMP
  #pragma omp parallel for 
#endif
  for(c=0; c < seg->width; c++) {
    int r,s;
    int val;
    for(r=0; r < seg->height; r++) {
      for(s=0; s < seg->depth; s++) {
	val = MRIgetVoxVal(wm,c,r,s,0);
	if(val != 1) continue;
	val = MRIgetVoxVal(seg,c,r,s,0);
	if(val ==  2) MRIsetVoxVal(outseg,c,r,s,0,5001);
	if(val == 41) MRIsetVoxVal(outseg,c,r,s,0,5002);
      }
    }
  }
  
  MRIfree(&wm);
  return(outseg);
}

