// Repository: TheGoldLab/Lab_Rex
// File: sset/long3_notworking.d

/*	long3.d
**		modified from long2.d to add color cues for asymmetric reward trials
** 	04-19-2012 added option (by setting color_cue to 2) to signal block change with color cues in the first trial after block change
** 
**
** 	paradigm for Long's use. Includes dot RT task, regular Mem sac task, 
**	vis sac task
**	Asymmetric reward versions accomplished by setting big/small reward, 
**
**	created by Long 1/08/07
** 3-20-2007 revised event sequence, Long Ding
** 3-13-2007 Long added regular dots task, wrong timing events
** Note 2-27-2007 Long: 
**		for ASL calibration and validation, reward is given by function 
**		pr_give_reward, which get reward size information from menu Preferences
**		menu items (Reward_on_time, Reward_off_time, Beep_every_reward); 
**		For MGS, VGS and dotsRT tasks, reward DIO is set directly in states by calling
**		fun_rewardon and fun_rewardoff. The time of state "rewardon" is set
**		for each trial in start_xxx functions, based on reward contingency and big/small
**		reward size in task property menus.
** running line convention: 0 - intertrial  5 - initial fp on, 10 - fp change after switching to task
**		20 - cue on (tgt on for dots)	 30 - dots on	40 - go	50 - sac	60 - rew
**	Added feature to get rough measure of RT (time from dots on to SACMAD ). Long 3-29-07
**	modified to accomodate changes in paradigm_rec.c Long 9-13-07
**		use the new subroutine in paradigm_rec.c to give reward for both ASL and other tasks.
**	  	no longer set reward DIO in states
**		
**	Long Ding 2008-04-16. before today, several ecodes were not dropped because dx_check function was not called.
** 		modified to drop TRGACQUIREDCD and FDBKONCD. For data collected before, use SACMAD as an equivalent event 
**  			plus appropriate delays
**	Long Ding 2008-09-10	Added the option to withheld reward in easy trials. randomly omit reward in 
**			VGS, MGS, and high coh dots trials
** Long Ding 2009-05-19 changed Rew_by_RT function to separately set slope for the two directions
** Long Ding 2009-06-04 added option to deliver estim during dots viewing in dotsRT task only
** Long Ding 2009-06-30 changed dotsreg task to repeat previous viewing time for fixbreak or error+flagRepeat
** Long Ding 2010-10-22 added another msg task for FEF estim confirmation. Fixoff-gap-estim to make sure monkey's eye is roughly at center of screen before estim
** Long Ding 2013-01-04 added a dotsardt task
## Long Ding 2013-02-19 revised color cue presentation. Now use CLUT 8(small) and 9(large) for asymmetric reward tasks
*/

#include "rexHdr.h"
#include "paradigm_rec.h"
#include "toys.h"
#include "lcode.h"

/* PRIVATE data structures */

/* GLOBAL VARIABLES */
static _PRrecord 	gl_rec = NULL; /* KA-HU-NA */
long currentrew= 0;
long totalrew = 0;
long timego = 0;
long timesac = 0;
int flag_omitreward = 0;
int flag_asymRew = 0;
int flag_blockchange = 0;
int flag_estim = 0;
int flagTrialRepeat = 0;
int prevtime = 10; /* for tracking viewing time used in the previous error or fixbreak trial */

	/* for now, allocate these here... */
MENU 	 	 umenus[200];
RTVAR		 rtvars[15];
USER_FUNC ufuncs[15];

/* MACROS for memory+visual sac task */

#define TTMV(n) 	pl_list_get_v(gl_rec->trialP->task->task_menus->lists[0],(n))
#define TPMV(n) 	pl_list_get_v(gl_rec->trialP->task->task_menus->lists[1],(n))

#define WIND0 0		/* window to compare with eye signal */
#define WIND1 1		/* window for correct target */
#define WIND2 2		/* window for other target in dots task */
	/* added two dummy window to signal task events in rex window */
#define WIND3 3 		/* dummy window for fix point */
#define WIND4 4 		/* dummy window for target */
#define EYEH_SIG 	0
#define EYEV_SIG 	1

#define FPCOLOR_INI	15
#define FPCOLOR_ASL	1
#define FPCOLOR_MGS	2
#define FPCOLOR_VGS	3
#define FPCOLOR_DOT	4
#define FPCOLOR_REGDOT 1
#define REWINDEX		0
#define ANGLEINDEX	1
#define REWCUE_LARGE 9
#define REWCUE_SMALL 8

#define STIMDIO PR_DIO_ID(5) /* input pulse to Grass */

/* ROUTINES */

/*
**** INITIALIZATION routines
*/
/* ROUTINE: autoinit
**
**	Initialize gl_rec. This will automatically
**		set up the menus, etc.
**	Long's note: the number after ufuncs determines how many sets of menus each task gets
*/
void autoinit(void)
{

// printf("autoinit start\n");

	gl_rec = pr_initV(0, 0, 
		umenus, NULL,
		rtvars, pl_list_initV("rtvars", 0, 1, 
				"currentrew", 0, 1.0,
				"totalrew", 0, 1.0,	
				"omitrew", 0, 1.0, 
				 NULL),
		ufuncs, 
		"asl", 1, 
		"mgs", 1, 
		"vgs",  1,
		"dotsrt", 1,
		"dotsreg", 1,
		"mgs", 1,
		"dotsardt", 1,
		NULL);

// printf("autoinit end\n");
}

/* ROUTINE: rinitf
**
** initialize at first pass or at r s from keyboard 
*/
void rinitf(void)
{
	static int first_time = 1;

	/* This stuff needs to be done only once, but also
	**		needs to be done after the clock has started
	**		(so do NOT put it up in autoinit).
	*/
	if(first_time) {

		/* do this once */
		first_time = 0;

		/* initialize interface (window) parameters */
		wd_src_pos(WIND0, WD_DIRPOS, 0, WD_DIRPOS, 0);
		wd_src_check(WIND0, WD_SIGNAL, EYEH_SIG, WD_SIGNAL, EYEV_SIG);

		wd_src_pos(WIND1, WD_DIRPOS, 0, WD_DIRPOS, 0);
		wd_src_check(WIND1, WD_SIGNAL, EYEH_SIG, WD_SIGNAL, EYEV_SIG);

		wd_src_pos(WIND2, WD_DIRPOS, 0, WD_DIRPOS, 0);
		wd_src_check(WIND2, WD_SIGNAL, EYEH_SIG, WD_SIGNAL, EYEV_SIG);

		wd_src_pos(WIND3, WD_DIRPOS, 0, WD_DIRPOS, 0);
		wd_src_check(WIND3, WD_SIGNAL, EYEH_SIG, WD_SIGNAL, EYEV_SIG);

		wd_src_pos(WIND4, WD_DIRPOS, 0, WD_DIRPOS, 0);
		wd_src_check(WIND4, WD_SIGNAL, EYEH_SIG, WD_SIGNAL, EYEV_SIG);

		/* init the screen */
		pr_setup();
	}
}

/* ROUTINE: start_trial
**
*/
int start_trial(void)
{
	int task_index;

	if(!pr_start_trial()) {
		pr_toggle_file(0);
		return(0);
	}

 	task_index = pr_get_task_index();
   set_times("fpshow_delay", 1, -1); 
	if (task_index>0)
	{
      printf(" flag_asymRew %d  REWINDEX %d \n", flag_asymRew, PL_L2PV(gl_rec->trialP->list, REWINDEX) );
		if ( ( flag_asymRew!=PL_L2PV(gl_rec->trialP->list, REWINDEX) ) && (task_index!=6) )
		{
			set_times("fpshow_delay", TTMV("inter_block"), -1);	
			flag_asymRew = PL_L2PV(gl_rec->trialP->list, REWINDEX);
			flag_blockchange = 1;
		}
	}
	/* No dynamic stimuli, so make draw_flag=3 the default.
	**	This draws each command ONCE
	*/
	/* dx_set_flags: (in dotsx.c)
	**
 	
	dx_set_flags(DXF_D3);
	*/
	
	dx_set_flags(DXF_D1);
	wd_siz(WIND3, 0, 0);
	wd_siz(WIND4, 0, 0);

	return(MET);
}



int show_error(long error_type)
{

	if(error_type == 0)
		printf("Bad task index (%d)\n", pr_get_task_index());

	return(0);
}

/* ROUTINE: start_mgs
**
**	set times for states and set reward info used in memory sac task 
*/
int start_mgs( void )
{
	int smallrew, bigrew;
	int angle0, angle_cur;
	int p_omit;	/* prob of omitting reward in correct trials, in percent */

	long tarwin;

	/* set basic task timing */

	set_times("precue", TTMV("precue"), -1);
	set_times("delay", TTMV("delay"), -1);
	set_times("waitforsac", TTMV("wait4sacon"), -1);
	set_times("sacon", TTMV("wait4hit"), -1);
	set_times("tgtacq", TTMV("gap2feedback"), -1);
	set_times("rewarddelay", TTMV("delay2rew"), -1);
	
	/* set reward size based on current trial target loc and rew contingency */

	smallrew = TPMV("smallreward");
	bigrew = TPMV("bigreward");
	flag_asymRew = PL_L2PV(gl_rec->trialP->list, REWINDEX);
	if ( flag_asymRew == 2) {
		/* equal reward */
		currentrew = (smallrew+bigrew)/2;
	} else {
		/* asymmetric reward, assumes 2 possible targets only */
		angle_cur = PL_L2PV(gl_rec->trialP->list, ANGLEINDEX);
		angle0 = TPMV("Angle_o");
		if (flag_asymRew) {
			currentrew = smallrew;
			if (angle_cur == angle0) currentrew = bigrew;
		} else {
			currentrew = bigrew;
			if (angle_cur == angle0) currentrew = smallrew;
		}
	}
/*	set_times("rewardon", currentrew, -1); */ 

   /* change target color according to reward if ColorCue is set to 1 */
   if ( (flag_asymRew == 2) || (TPMV("ColorCue")==0) ) {
     dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
         DX_DIAM, 5,
         DX_CLUT, 1, NULL);
   } else {
      if (currentrew == smallrew)
      dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
         DX_DIAM, 5,
         DX_CLUT, REWCUE_SMALL, NULL); /* BLUE */
      if (currentrew == bigrew)
      dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
         DX_DIAM, 5,
         DX_CLUT, REWCUE_LARGE, NULL); /* GREEN */
   }

	/* turn on the target window */
	tarwin = TPMV("targetwin");
	dx_position_window(tarwin, tarwin, 1, 0, 1);
	dx_position_window(tarwin, tarwin, 1, 0, 2);
	/* turn on the dummy window on rex to indicate fix on */
	dx_position_window(10, 10, 0, 0, 3);
	/* start checking fixation break */
	dx_set_fix(1); 
	
	/* decide if to omit reward */
	p_omit = TPMV("OmitRew") * 10;
	flag_omitreward = TOY_RCMP(p_omit);

   /* set rtvars */
   pr_set_rtvar("currentrew", currentrew);
   pr_set_rtvar("omitrew", flag_omitreward);

	return(0);
}

/* ROUTINE: start_vgs
**
**	set times for states and set reward info used in visually-guided sac task 
*/
int start_vgs(void)
{

 	int smallrew, bigrew;
	int angle0, angle_cur;
   int p_omit; /* prob of omitting reward in correct trials, in percent */

	long tarwin;

	/* set basic task timing */

	set_times("pretgt", TTMV("pretgt"), -1);
	set_times("waitforsac_vgs", TTMV("wait4sacon"), -1);
	set_times("getsactime_vgs", TTMV("wait4hit"), -1);
	set_times("rewarddelay_vgs", TTMV("delay2rew"), -1);
	
	/* set reward size based on current trial target loc and rew contingency */

	smallrew = TPMV("smallreward");
	bigrew = TPMV("bigreward");
	flag_asymRew = PL_L2PV(gl_rec->trialP->list, REWINDEX);
	if ( flag_asymRew == 2) {
		/* equal reward */
		currentrew = (smallrew+bigrew)/2;
	} else {
		/* asymmetric reward, assumes 2 possible targets only */
		angle_cur = PL_L2PV(gl_rec->trialP->list, ANGLEINDEX);
		angle0 = TPMV("Angle_o");
		if (flag_asymRew) {
			currentrew = smallrew;
			if (angle_cur == angle0) currentrew = bigrew;
		} else {
			currentrew = bigrew;
			if (angle_cur == angle0) currentrew = smallrew;
		}
	}
/*	set_times("rewardon", currentrew, -1); */

   /* change target color according to reward if ColorCue is set to 1 */
   if ( (flag_asymRew == 2) || (TPMV("ColorCue")==0) ) {
     dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL, 
			DX_DIAM, 5, 	
			DX_CLUT, 1, NULL);
   } else {
      if (currentrew == smallrew)
		dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
			DX_DIAM, 5,
			DX_CLUT, REWCUE_SMALL, NULL); /* BLUE */
      if (currentrew == bigrew) 
		dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
			DX_DIAM, 5,
			DX_CLUT, REWCUE_LARGE, NULL); /* GREEN */
   }

	/* turn on the target window */
	tarwin = TPMV("targetwin");
	dx_position_window(tarwin, tarwin, 1, 0, 1);

	/* turn on the dummy window on rex to indicate fix on */
	dx_position_window(10, 10, 0, 0, 3);
   /* start checking fixation break */
   dx_set_fix(1);
	
 
   /* decide if to omit reward */
   p_omit = TPMV("OmitRew") * 10;
   flag_omitreward = TOY_RCMP(p_omit);

   /* set rtvars */
   pr_set_rtvar("currentrew", currentrew);
   pr_set_rtvar("omitrew", flag_omitreward);

	return(0);
}

/* ROUTINE: start_dots
**
**	set times for states and set reward info used in dots task 
*/
int start_dots( void )
{
	int smallrew, bigrew;
	int angle0, angle_cur;
	int coh, maxcoh;
   int p_omit; /* prob of omitting reward in correct trials, in percent */
	int p_estim;

	long tarwin;
	/* set basic task timing */

	set_times("waitforsac_dots", TTMV("wait4sacon"), -1);
	set_times("wait4hit_dots", TTMV("wait4hit"), -1);

/*	set_times("rewarddelay_dots", TTMV("delay2rew"), -1); */

	set_times("error_dots", TTMV("errorFB"), -1);


	/* set reward size based on current trial target loc and rew contingency */

	smallrew = TPMV("smallreward");
	bigrew = TPMV("bigreward");
	flag_asymRew = PL_L2PV(gl_rec->trialP->list, REWINDEX);
	if ( flag_asymRew == 2) {
		/* equal reward */
		currentrew = (smallrew+bigrew)/2;
	} else {
		/* asymmetric reward, assumes 2 possible targets only */
		angle_cur = PL_G2PVS(gl_rec->dx->current_graphic, 1, 6, 0);
		angle0 = TPMV("Angle_o");
		if (flag_asymRew) {
			currentrew = smallrew;
			if (angle_cur == angle0) currentrew = bigrew;
		} else {
			currentrew = bigrew;
			if (angle_cur == angle0) currentrew = smallrew;
		}
		
	}
/*	set_times("rewardon", currentrew, -1); */ 

   /* change target color according to reward if ColorCue is set to 1 */
printf("flag_blockchange %d \n", flag_blockchange); 
   if ( (flag_asymRew == 2) 
			|| (TPMV("ColorCue")==0) 
			|| ( (TPMV("ColorCue")==2) && (flag_blockchange==0) ) 
		) {
     dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
         DX_DIAM, 5,
         DX_CLUT, 1, NULL);
     dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 2, NULL,
         DX_DIAM, 5,
         DX_CLUT, 1, NULL);
   } else {
      if (currentrew == smallrew) {
      	dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
      	   DX_DIAM, 5,
      	   DX_CLUT, REWCUE_SMALL, NULL); /* BLUE */
      	dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 2, NULL,
      	   DX_DIAM, 5,
      	   DX_CLUT, REWCUE_LARGE, NULL); /* GREEN */
		} else {
      	dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
      	   DX_DIAM, 5,
      	   DX_CLUT, REWCUE_LARGE, NULL); /* GREEN */
         dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 2, NULL,
      	   DX_DIAM, 5,
      	   DX_CLUT, REWCUE_SMALL, NULL); /* BLUE */
		}
	}
	
	/* turn on the target window */
	tarwin = TPMV("targetwin");
	dx_position_window(tarwin, tarwin, 1, 0, 1);

	dx_position_window(tarwin, tarwin, 2, 0, 2);

	/* turn on the dummy window on rex to indicate fix on */
	dx_position_window(10, 10, 0, 0, 3);
   /* start checking fixation break */
   dx_set_fix(1);
	
  
	/* decide if to omit reward */
 	/* self note: usage of PL_G2PVS: 1: list 1 for dots targets
												5:	property #5, as defined in dotsX.c
												0: the first dots target 
	*/

  	coh = PL_G2PVS( gl_rec->dx->current_graphic, 1, 5, 0);
   maxcoh = TPMV("Coherence_hi");
   if (coh == maxcoh) {
		p_omit = TPMV("OmitRew") * 10;
  	 	flag_omitreward = TOY_RCMP(p_omit);
	}
	else {
		flag_omitreward = 0;
	}

	/* decide if to deliver electrical stimulation */
	p_estim = TPMV("prob_estim");	
	flag_estim = 0;	
	if (p_estim>0)
		flag_estim = TOY_RCMP(p_estim * 10);
	
   /* set rtvars */
   pr_set_rtvar("currentrew", currentrew);
   pr_set_rtvar("omitrew", flag_omitreward);
	
	return(0);
}


/* ROUTINE: start_dotsreg
**
**	set times for states and set reward info used in regular dots task 
*/
int start_dotsreg( void )
{
	int smallrew, bigrew;
	int angle0, angle_cur;

	long tarwin;
	/* set basic task timing */

	set_times("waitforsac_dotsreg", TTMV("wait4sacon"), -1);
	set_times("wait4hit_dotsreg", TTMV("wait4hit"), -1);

	set_times("rewarddelay_dotsreg", TTMV("delay2rew"), -1);

	set_times("bothtgtoffreg", TTMV("errorFB"), -1); 

	/* set reward size based on current trial target loc and rew contingency */

	smallrew = TPMV("smallreward");
	bigrew = TPMV("bigreward");
	flag_asymRew = PL_L2PV(gl_rec->trialP->list, REWINDEX);
	if ( flag_asymRew == 2) {
		/* equal reward */
		currentrew = (smallrew+bigrew)/2;
	} else {
		/* asymmetric reward, assumes 2 possible targets only */
		angle_cur = PL_G2PVS(gl_rec->dx->current_graphic, 1, 6, 0);
		angle0 = TPMV("Angle_o");
		if (flag_asymRew) {
			currentrew = smallrew;
			if (angle_cur == angle0) currentrew = bigrew;
		} else {
			currentrew = bigrew;
			if (angle_cur == angle0) currentrew = smallrew;
		}
	}
/*	set_times("rewardon", currentrew, -1); */

	/* turn on the target window */
	tarwin = TPMV("targetwin");
	dx_position_window(tarwin, tarwin, 1, 0, 1);

	dx_position_window(tarwin, tarwin, 2, 0, 2);

	/* turn on the dummy window on rex to indicate fix on */
	dx_position_window(10, 10, 0, 0, 3);
   /* start checking fixation break */
   dx_set_fix(1);
	/* set rtvars */
	pr_set_rtvar("currentrew", currentrew);


	return(0);
}


int set_dotsreg_viewtime(void)
{
	int exptime = TTMV("exp_time");
	int mintime = TTMV("min_time");
	int maxtime = TTMV("max_time");
	int meantime = TTMV("mean_time");
	int override = TTMV("override");
	int override_random = TTMV("override_random");

	if (flagTrialRepeat)
	{
      prevtime = timer_set1(0,0,0,0,prevtime,0);
	}
	else
	{
		prevtime = timer_set1(exptime, mintime,maxtime, meantime, override, override_random);
	}
printf(" viewing time %d \n", prevtime);
	flagTrialRepeat = 1;
	return(0);
}


/* ROUTINE: start_dotsardt
**
** set times for states and set reward info used in dots task
*/
int start_dotsardt( void )
{
   int smallrew, bigrew;
   int angle0, angle_cur;
   int coh, maxcoh;
   int p_estim;

   long tarwin;
   /* set basic task timing */

   set_times("waitforsac_ARdt", TTMV("wait4sacon"), -1);
   set_times("wait4hit_ARdt", TTMV("wait4hit"), -1);
   set_times("error_ARdt", TTMV("errorFB"), -1);

   /* set reward size based on current trial target loc and rew contingency */

   smallrew = TPMV("smallreward");
   bigrew = TPMV("bigreward");
   flag_asymRew = PL_L2PV(gl_rec->trialP->list, REWINDEX);
   if ( flag_asymRew == 2) {
      /* equal reward */
      currentrew = (smallrew+bigrew)/2;
   } else {
      /* asymmetric reward, assumes 2 possible targets only */
      angle_cur = PL_G2PVS(gl_rec->dx->current_graphic, 1, 6, 0);
      angle0 = TPMV("Angle_o");
      if ( (flag_asymRew == 1)
            || (flag_asymRew == 4) ) {
         currentrew = smallrew;
         if (angle_cur == angle0) currentrew = bigrew;
      } else {
         currentrew = bigrew;
         if (angle_cur == angle0) currentrew = smallrew;
      }

   }
   set_times("ar_cue_delay", TPMV("dT_AR"), -1);

   /* change target color according to flag_asymRew */
   if (flag_asymRew >= 2) {
     dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
         DX_DIAM, 5,
         DX_CLUT, 1, NULL);
     dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 2, NULL,
         DX_DIAM, 5,
         DX_CLUT, 1, NULL);
   } else {
      if (currentrew == smallrew) {
         dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
            DX_DIAM, 5,
            DX_CLUT, REWCUE_SMALL, NULL); /* BLUE */
         dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 2, NULL,
            DX_DIAM, 5,
            DX_CLUT, REWCUE_LARGE, NULL); /* GREEN */
      } else {
         dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
            DX_DIAM, 5,
            DX_CLUT, REWCUE_LARGE, NULL); /* GREEN */
         dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 2, NULL,
            DX_DIAM, 5,
            DX_CLUT, REWCUE_SMALL, NULL); /* BLUE */
      }
   }
   /* turn on the target window */
   tarwin = TPMV("targetwin");
   dx_position_window(tarwin, tarwin, 1, 0, 1);
   dx_position_window(tarwin, tarwin, 2, 0, 2);

   /* turn on the dummy window on rex to indicate fix on */
   dx_position_window(10, 10, 0, 0, 3);
   /* start checking fixation break */
   dx_set_fix(1);


   /* self note: usage of PL_G2PVS: 1: list 1 for dots targets
                                    5: property #5, as defined in dotsX.c
                                    0: the first dots target
   */

   /* decide if to deliver electrical stimulation */
   p_estim = TPMV("prob_estim");
   flag_estim = 0;
   if (p_estim>0)
      flag_estim = TOY_RCMP(p_estim * 10);

   /* set rtvars */
   pr_set_rtvar("currentrew", currentrew);

   return(0);
}

/* ROUTINE: ar_ue_change
** check to see if need to change target colors according to flag_asymRew
*/
int ar_cue_change(void)
{
   int smallrew, bigrew;

   smallrew = TPMV("smallreward");
   bigrew = TPMV("bigreward");
   /* change target color according to flag_asymRew */
   if (flag_asymRew > 2) {
		ec_send_hi(QONCD);
      if (currentrew == smallrew) {
         dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
            DX_DIAM, 5,
            DX_CLUT, REWCUE_SMALL, NULL); /* BLUE */
         dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 2, NULL,
            DX_DIAM, 5,
            DX_CLUT, REWCUE_LARGE, NULL); /* GREEN */
      } else {
         dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 1, NULL,
            DX_DIAM, 5,
            DX_CLUT, REWCUE_LARGE, NULL); /* GREEN */
         dx_set_by_nameIV(DXF_DEFAULT, DX_TARGET, 2, NULL,
            DX_DIAM, 5,
            DX_CLUT, REWCUE_SMALL, NULL); /* BLUE */
      }
   }
	return(0);
}


/* ROUTINE: do_calibration
**
**	Returns 1 if doing calibration (ASL task)
*/
int do_calibration(void)
{
	if( pr_get_task_index() == 0 && 
		pr_get_task_menu_value("setup", "Cal0/Val1", 0) == 0) {
		return(1);
	} else {
		return(0);
	}
}

int dummycue(void)
{
	/* turn on the dummy window on rex to indicate cue on */
	dx_position_window(5, 5, 0, 1, 4);
	return(0);
}

int dummyfpcueoff(void)
{
	int a = pr_get_task_index();
	dx_position_window(0, 0, 0, 0, 3);
	dx_position_window(0, 0, 0, 0, 4);
	if (a == 1) {
		timego = getClockTime();
	}
	return(0);
}

int dummytgt(void)
{
	/* turn on the dummy window on rex to indicate tgt on, also turn off dummy fp win */
	dx_position_window(5, 5, 1, 1, 4);
	dx_position_window(0, 0, 0, 1, 3);
	return(0);
}

int dummydots(void)
{
	dx_position_window(4, 4, 0, 0, 3);
	return(0);
}

int dotsonFPchange(void)
{

	int dxtarget_indices[1] = {0};
	
	/* change FP CLUT value to menu CLUT#14 */

	dx_set_by_indexIV(DXF_D1, 0, 1, dxtarget_indices, 4, 14, ENDI);
	
	/* addded by Long, 03-29-07, get clock time for rough measure of RT */
	
	timego = getClockTime();
	
	/* added by Long, 3-30-07, set timer for minimal delay to reward */

	timer_set1(0, 0, 0, 0, TTMV("delay2rew"), 0);
	timer_set2(0, 0, 0, 0, TTMV("delay2rew") + currentrew + TTMV("errorFB"), 0);

	return(0);
}

int tgton_time(void)
{
   timego = getClockTime();
	return(0);
}

int fun_rewardon(long code)
{
	valtype dv;
	if (code>0)
	 	ec_send_code(code);
	if((dv=DIV("Reward_bit"))>=0) 
	{
		dio_on(PR_DIO_ID(dv));
		ec_send_dio(dv);
	}
	return(0);
/*	 PR_DIO_ON("Reward_bit")
;	*/
}


int fun_rewardoff(long code)
{
	valtype dv;
	if (code>0)
	 	ec_send_code(code);
	if((dv=DIV("Reward_bit"))>=0) 
	{
		dio_off(PR_DIO_ID(dv));
		ec_send_dio(dv);
	}
	return(0);
/*	 PR_DIO_OFF("Reward_bit");	*/
}

int getsactime(void)
{
	timesac = getClockTime();
}

int set_reward(void)
{
	int i = 0;
	int rewnum = 0;
	i = pr_get_task_index();
	if (i==40)
	{	
   	rewnum = TPMV("reward_num");
		pr_set_reward(rewnum, currentrew, pl_list_get_v(gl_rec->prefs_menu, "Reward_off_time"), -1, 0, 0);
		totalrew = totalrew + rewnum * currentrew;
	}
	else
	{ 

		pr_set_reward(1, currentrew, pl_list_get_v(gl_rec->prefs_menu, "Reward_off_time"), -1, 0, 0);
		totalrew = totalrew + currentrew;
	} 
  	pr_set_rtvar("totalrew", totalrew);
	return(0);
}

int correctfun(void)
{
		printf(" %d ", timesac-timego);
      // return( pr_score_trialRT(kCorrect, currentrew, 0, timesac-timego, 0, 0, 1) );
		pr_score_trialRT(kCorrect, 0, 1, timesac-timego);
		// pr_set_reward(1, currentrew, 1, -1, 0, 0); 
		flagTrialRepeat = 0;
		set_times("extra_delay_dots", 1 ,-1 );
		flag_blockchange = 0;
		return(0);

}


int errorfun(void)
{
	int delay_by_RT = 1;
 	printf("error RT %d ", timesac-timego);
	pr_score_trial(kError, 0, 1);
	flagTrialRepeat = 0;
	if (TPMV("flagRepeat")==1)
		flagTrialRepeat = 1;
/*	if ( (TPMV("flagRepeat")==2 )
			&& ( (timesac-timego)<800 )
			&& ( abs( PL_L2PV(gl_rec->trialP->list, ANGLEINDEX) )<100 ) )
		flagTrialRepeat = 1;
*/
	if (timesac-timego<1000)	
 		set_times("extra_delay_dots", (TPMV("extra_delayRT")-timesac+timego)*TPMV("extra_delayslope"),-1 );
	flag_blockchange = 0;

}


int rew_by_RT( long slope1, long baseRT1, long slope2, long baseRT2, long minRew, long maxRew)
{
/* note: input slope should be 10 times the intended slope */

	double baseRT, tempRew;
	int flag_asymRew;
	int angle0, angle_cur;	
	int slope;

	flag_asymRew = PL_L2PV(gl_rec->trialP->list, REWINDEX);

	if ( flag_asymRew==2 && slope1+slope2>0 ) 
	{	
	  	angle_cur = PL_G2PVS(gl_rec->dx->current_graphic, 1, 6, 0);
     	angle0 = TPMV("Angle_o");
		if (angle_cur == angle0)
		{
			baseRT = baseRT1;
			slope  = slope1;
		}
		else
      {
         baseRT = baseRT2;
			slope  = slope2;
		}
		tempRew = (double)currentrew + (timesac - timego - baseRT) * slope / 10.0;		
		currentrew = (int)tempRew;
		if (tempRew<minRew)	currentrew = minRew;
		if (tempRew>maxRew) 	currentrew = maxRew;
/*		set_times("rewardon", currentrew, -1); */
		pr_set_rtvar("currentrew", currentrew);
	}

	return(0);		

}


int fun_estim(long ecode, long flag)
{
   fun_dio(ecode, STIMDIO, flag);
   return(0);
}


int fun_dio(long ecode, DIO_ID id, long flag)
{
   if (flag==1)
   {
      dio_on(id);
   }
   else
   {
      dio_off(id);
   }
   if (ecode>0)
   {
      ec_send_code(ecode);
      ec_send_dio(id);
   }
   return(0);
}


/* THE STATE SET 
*/
%%
id 1000
restart rinitf
main_set {
status ON
begin	first:
		to prewait
   /*
   ** First wait time, which can be used to add
   ** a delay after reset states (e.g., fixation break)
   ** timer_set: (in timerLT.c) a function to set a timer from an exponential distribution
   **		of values. 5 such timers are currently available. 
   ** 	e.g., timer_set1 sets the first timer 
   */
	prewait:
		do timer_set1(0,100,600,200,0,0)
		to loop on +MET % timer_check1
   /*
   ** Start the loop!
   ** Note that the real workhorse here is pr_next_trial,
   **    called in go. It calls the task-specific
   **    "get trial" and "set trial" methods,
   **    and drops STARTCD and LISTDONECD, as
   **    appropriate.
   **
   */
	loop:
		time 1000
		to pause on +PSTOP & softswitch
		to go
	pause:
		to go on -PSTOP & softswitch
	/*
    ** pr_start_trial: (in paradigm_rec.c) run get_trial and send start ecodes and task and trial ecodes to PLEXON 
    ** regular task event ecodes are not tagged. Task and trial ecodes are tagged, send first the tag then the value. 
    */
	go:
		do pr_toggle_file(1)
		to trstart
	trstart:
		to fpshow_delay on MET % start_trial
		to loop	
	fpshow_delay:
		to fpshow
	fpshow:
		do dx_show_fp(FPONCD, 0, 3, 3, FPCOLOR_INI, FPCOLOR_INI)		
		rl 5
		to caljmp on DX_MSG % dx_check
	
	caljmp:
		to calstart on 1 % do_calibration
		to fpwinpos

	/* CALIBRATION TASK
	** Check for joystick button press indicating a correct fixation
	** missed targets are scored as NC in order to be shown again later
	*/
	calstart:
		time 5000
		to calacc on 0 % dio_check_joybut
		to ncerr
	calacc:
		time 10
		do ec_send_code(ACCEPTCAL) 
		to correctASL

	fpwinpos:
		time 20  /* takes time to settle window */
		do dx_position_window(60, 60, 0, 0, 0)
 		to fpwait
	fpwait:
 		time 5000
		to fpset on -WD0_XY & eyeflag
		to fpnofix
	fpnofix:    /* failed to attain fixation */
		time 2500
		do pr_score_trial(kNoFix,0,1)
		to finish
	fpset:
		time 50 /* give gaze time to settle into place (fixation) */
		do ec_send_code(EYINWD)
		to fpwait on +WD0_XY & eyeflag
		to fpwin2
	fpwin2:
		time 20 /* again time to settle window */
		do dx_position_window(40, 40, 0, 1, 0)
		to taskjmp

	/* Jump to task-specific statelists
	*/
	taskjmp:
		to t1fp on 0 % pr_get_task_index	
		to t2fp on 1 % pr_get_task_index
		to t3fp on 2 % pr_get_task_index	
		to t4fp on 3 % pr_get_task_index	
		to t5fp on 4 % pr_get_task_index	
		to t6fp on 5 % pr_get_task_index 
		to t7fp on 6 % pr_get_task_index
		to badtask
	badtask:
		do show_error(0)
		to finish

	/* TASK 1: calibrate the ASL eye tracker  */
	t1fp:
		do dx_show_fp(FPCHG, 0, 5, 5, FPCOLOR_ASL, FPCOLOR_ASL);
		rl 10
		to t1wait1 on DX_MSG % dx_check
	t1wait1:
		do timer_set1(1000, 100, 600, 200, 0, 0)
 		to t1winpos on MET % timer_check1
	t1winpos:
		time 20
		do dx_position_window(20, 20,-1,0,0)
 		to correctASL

	/* TASK 2: memory-guided saccade	*/
	t2fp:
		do dx_show_fp(FPCHG, 0, 3, 3,  FPCOLOR_MGS,  FPCOLOR_MGS)
		rl 10
		to start_mgs_state on DX_MSG % dx_check
	start_mgs_state:
		do start_mgs()
		to	precue 
	precue:
		to fixbreak on +WD0_XY & eyeflag
		to cueon
	cueon:
		do dx_toggle2(TARGONCD, 1, 0, 0, 1, 1000)
		rl 20
		to cuedur on DX_MSG % dx_check
	cuedur:
		time 100
		do dummycue()
		to fixbreak on +WD0_XY & eyeflag
		to cueoff
	cueoff:
		do dx_toggle2(TARGOFFCD, 0, 0, 0, 1, 1000)
		to delay on DX_MSG % dx_check
	delay:
		to fixbreak on +WD0_XY & eyeflag
		to fixoffgo
	fixoffgo:
		do dx_toggle2(FPOFFCD, 0, 0, 1000, 0, 1000)
		rl 40
		to waitforsac on DX_MSG % dx_check
	waitforsac:
		do dummyfpcueoff()
		to getsactime_mgs on +WD0_XY & eyeflag
		to errfeedback
   getsactime_mgs:
      do getsactime()
		to sacon
	sacon:
		do ec_send_code_lo(SACMAD)
		rl 50
		to tgtacq on -WD1_XY & eyeflag 		
		to errfeedback
	tgtacq:	
		do ec_send_code(TRGACQUIRECD)
		to fixbreak_cueon on +WD1_XY & eyeflag
		to cuefeedback
   fixbreak_cueon:
		time 50
      do dx_toggle2(FDBKONCD, 1, 0, 0, 1, 1000)
      to fixbreak 
	cuefeedback:
		do dx_toggle2(FDBKONCD, 1, 0, 0, 1, 1000)
		to rewarddelay on DX_MSG % dx_check
	rewarddelay:
/* added omit_reward 9-10-2008 Long Ding */
		to omit_reward
	omit_reward:
		to omitecode on +1 & flag_omitreward
		to rewardon  
	omitecode:
		do ec_send_code_hi(OMITREW)
		to correct

/* implemented 9-14-07 Long, using the new subroutines in paradigm_rec.c */ 
	rewardon:
		do set_reward()
		to correct on 0 % pr_beep_reward

/* for the UW water feeder, toggle method for computer control */
/* 
	rewardon:
		rl 60
		do fun_rewardon(REWCD)
		to rewbitoff1
	rewbitoff1:
		do fun_rewardoff(0)
		to rewbiton2
	rewbiton2:
		do fun_rewardon(REWOFFCD)
		to rewardoff
	rewardoff:
		do fun_rewardoff(0)
		to correct
*/

/* for the Crist reward box, follower method for computer control */
/*	rewardon:
		do fun_rewardon(REWCD)	
		to rewardoff
	rewardoff:
		do fun_rewardoff(REWOFFCD)
		to correct
*/		
		
	errfeedback:
		do dx_toggle2(FDBKONCD, 1, 0, 0, 1, 1000)
		to errfeedbackoff on DX_MSG % dx_check
	errfeedbackoff:
		time 100
		to error 


	/* TASK 3: visually-guided saccade	*/
	t3fp:
		do dx_show_fp(FPCHG, 0, 3, 3,  FPCOLOR_VGS,  FPCOLOR_VGS)
		rl 10
		to start_vgs_state on DX_MSG % dx_check
	start_vgs_state:
		do start_vgs()
		to pretgt
	pretgt:
//		to fixbreak on +WD0_XY & eyeflag
		to tgton
	tgton:
		do dx_toggle2(TARGONCD, 2, 0, 0, 1, 1000)
		rl 40
		to getgotime_vgs on DX_MSG % dx_check
	getgotime_vgs:
		do tgton_time()
		to refrain_vgs 
	refrain_vgs:
		time 50
		do dummytgt()
//		to fixbreak on +WD0_XY & eyeflag
		to waitforsac_vgs	
	waitforsac_vgs:
		do dx_set_fix(0)
		to sacon_vgs on +WD0_XY & eyeflag
		to ncerr
	sacon_vgs:
		do ec_send_code_lo(SACMAD)
		rl 50
		to getsactime_vgs
	getsactime_vgs:
     	do getsactime()
		to tgtacq_vgs on -WD1_XY & eyeflag
 		to errfeedback
	tgtacq_vgs:	
		time 150
		do ec_send_code(TRGACQUIRECD)
		to fixbreak on +WD1_XY & eyeflag
		to rewarddelay_vgs
	rewarddelay_vgs:
		to omit_reward 


	/* TASK 4: dots RT */
	t4fp:
		do dx_show_fp(FPCHG, 0, 3, 3,  FPCOLOR_DOT,  FPCOLOR_DOT)
		rl 10
		to start_dots_state on DX_MSG % dx_check
	start_dots_state:
		do start_dots()
		to t4wait1 
	t4wait1:
		do timer_set1(0, 0, 0, 0, 500, 0)
//		to fixbreak on +WD0_XY & eyeflag
		to target2on on MET % timer_check1
	target2on:
		do dx_toggle2(TARGONCD, 1, 1, 1000, 2, 1000)
		rl 20
		to dummytgton on DX_MSG % dx_check
	dummytgton:	
		do dummycue()
		to t4wait2
	t4wait2:
		do timer_set1(1000, 200, 3000, 500, 0, 0) 
		to fixbreak on +WD0_XY & eyeflag
 		to dotson on MET % timer_check1
	dotson:
		do dx_toggle2(GORANDCD, 1, 0, 1000, 3, 1000)
		rl 30
		to fpafterdots_dots on DX_MSG % dx_check
	fpafterdots_dots:
		do dotsonFPchange()
		to refrain_dots on DX_MSG % dx_check
	refrain_dots:
		time 50
		do dummytgt()
		to fixbreak on +WD0_XY & eyeflag
		to waitforsac_dots
	waitforsac_dots:
		do dx_set_fix(0)		
		to sacon_dots on +WD0_XY & eyeflag
		to ncerr 
	sacon_dots:
		do dx_toggle2(SACMAD, 0, 0, 1000, 3, 1000)
		rl 50
		to wait4hit_dots on DX_MSG % dx_check
	wait4hit_dots:
		do getsactime()
		to correct_dots on -WD1_XY & eyeflag
		to incorrect_dots on -WD2_XY & eyeflag
		to ncerr
	correct_dots:
		do dx_toggle2(TRGACQUIRECD, 0, 1, 0, 2, 1000)
		to delay_correct_dots on DX_MSG % dx_check
	delay_correct_dots:
		time 400	
    	to ncerr on +WD1_XY & eyeflag
		to correcttgtfix 
	correcttgtfix:
	   do dx_toggle2(TARGOFFCD, 0, 1, 0, 2, 1000)	
		rl 60	
		to rewarddelay_dots on DX_MSG % dx_check	
	rewarddelay_dots:
		do rew_by_RT(20, 800, 0, 9000, 800, 1100)	
		to omit_reward on +MET % timer_check1
	incorrect_dots:
     	do dx_toggle2(TRGACQUIRECD, 0, 1, 1000, 2, 0)
		to delay_incorrect_dots on DX_MSG % dx_check
	delay_incorrect_dots:
		time 400 
   	to ncerr on +WD2_XY & eyeflag
		to errtgtfix 
	errtgtfix:
		do dx_toggle2(FDBKONCD, 2, 1, 1000, 2, 0)
		rl 60
		to delay_errtgtfix on DX_MSG % dx_check
	delay_errtgtfix:
		time 400
		to bothtgtoff_dots 
	bothtgtoff_dots:
		do dx_toggle2(TARGOFFCD, 0, 1, 1000, 2, 1000)	
		to error_dots on DX_MSG % dx_check
	error_dots:
		do errorfun()
		to finish_dots 
	finish_dots:
      do pr_finish_trial()
      rl 0
      to extra_delay_dots 
	extra_delay_dots:
		to wait2nexttrial

	/* TASK 5: regular dots */
	t5fp:
		do dx_show_fp(FPCHG, 0, 3, 3,  FPCOLOR_REGDOT,  FPCOLOR_REGDOT)
		rl 10
		to start_dotsreg_state on DX_MSG % dx_check
	start_dotsreg_state:
		do start_dotsreg()
		to t5wait1 
	t5wait1:
		do timer_set1(0,0,0,0,500,0)
		to fixbreak on +WD0_XY & eyeflag
		to tgtson_reg on MET % timer_check1
	tgtson_reg:
		do dx_toggle2(TARGONCD, 1, 1, 1000, 2, 1000)
		rl 20
		to dummytgt_dotsreg on DX_MSG % dx_check
	dummytgt_dotsreg:
		do dummycue()
		to delay2dots_reg 
	delay2dots_reg:
		do timer_set1(1000, 100, 600, 150, 0, 0)
		to fixbreak on +WD0_XY & eyeflag
		to dotson_reg on MET % timer_check1
	dotson_reg:
		do dx_toggle2(GORANDCD, 1, 0, 0, 3, 1000)
		rl 30
		to viewtimer on DX_MSG % dx_check
/*	dummydots_dotsreg:
		do dummydots()
		to viewtimer */
	viewtimer:
		do set_dotsreg_viewtime()
/*		do timer_set1(1000,100,1500,300,0,0) */
		to fixbreak on +WD0_XY & eyeflag
		to dotsoff_reg on MET % timer_check1
	dotsoff_reg:
		do dx_toggle2(ENDCD, 0, 0, 0, 3, 1000)
		to fixbreak on +WD0_XY & eyeflag
		to delay2fixoff_reg on DX_MSG % dx_check
	delay2fixoff_reg:
		do timer_set1(0, 0, 0, 0, 500, 0)
		to fixbreak on +WD0_XY & eyeflag
		to fixoff_reg on MET % timer_check1
	fixoff_reg:
		do dx_toggle2(FPOFFCD, 0, 0, 1000, 3, 0)
		rl 40
		to waitforsac_dotsreg on DX_MSG % dx_check
	waitforsac_dotsreg:
		do dummyfpcueoff()	
		to sacon_dotsreg on +WD0_XY & eyeflag
		to ncerr
	sacon_dotsreg:
		do ec_send_code_hi(SACMAD) 
		rl 50
		to wait4hit_dotsreg 
	wait4hit_dotsreg:
		to correct_dotsreg on -WD1_XY & eyeflag
		to incorrect_dotsreg on -WD2_XY & eyeflag
		to ncerr
	correct_dotsreg:
		time 150
		do ec_send_code_hi(TRGACQUIRECD)
		to fixbreak on +WD1_XY & eyeflag
		to rewarddelay_dotsreg
	rewarddelay_dotsreg:
		time 400
		to rewardon
	incorrect_dotsreg:
		time 150
		do ec_send_code_hi(TRGACQUIRECD)
		to fixbreak on +WD2_XY & eyeflag
		to errfeedback_dotsreg
	errfeedback_dotsreg:
		time 400
		do dx_toggle2(FDBKONCD, 0, 1, 0, 2, 1000)
		to bothtgtoff_dotsreg 
	bothtgtoff_dotsreg:
		do errorfun()
		to finish  

   /* TASK 6: fixation -> estim at fixoff */
   t6fp:
      do dx_show_fp(FPCHG, 0, 3, 3,  FPCOLOR_VGS,  FPCOLOR_VGS)
      rl 10
      to start_t6_state on DX_MSG % dx_check
   start_t6_state:
		time 500
      do start_mgs()
		to fixoff_t6
	fixoff_t6:
      do dx_toggle2(FPOFFCD, 0, 0, 1000, 0, 1000)
		rl 20
		to gap2estimon on DX_MSG % dx_check		
	gap2estimon:
		time 50
		to estimon
   estimon:
      time 5
      do fun_estim(STIMCD, 1)
      rl 30
      to estim_delay
   estim_delay:
      time 100
      to estimoff
   estimoff:
      do fun_estim(0, 0)
		rl 10
      to waitrew_t6
   waitrew_t6:
		time 500
      to rewardon

	 /* TASK 7: dots RT AR-dt */
	t7fp:
	   do dx_show_fp(FPCHG, 0, 3, 3,  FPCOLOR_DOT,  FPCOLOR_DOT)
	   rl 10
	   to start_dotsRT_ARdt_state on DX_MSG % dx_check
	start_dotsRT_ARdt_state:
	   do start_dotsardt()
	   to t7wait1
	t7wait1:
	   do timer_set1(0, 0, 0, 0, 500, 0)
 	   to target2on_ARdt on MET % timer_check1
	target2on_ARdt:
	   do dx_toggle2(TARGONCD, 1, 1, 1000, 2, 1000)
	   rl 20
	   to dummytgton_ARdt on DX_MSG % dx_check
	dummytgton_ARdt:
	   do dummycue()
	   to t7wait2
	t7wait2:
	   do timer_set1(1000, 200, 3000, 500, 0, 0)
	   to fixbreak on +WD0_XY & eyeflag
	   to dotson_ARdt on MET % timer_check1
	dotson_ARdt:
	   do dx_toggle2(GORANDCD, 1, 0, 0, 3, 1000)
	   rl 30
	   to fpafterdots_ARdt on DX_MSG % dx_check
	fpafterdots_ARdt:
	   do dotsonFPchange() 
	   to ar_cue_delay on DX_MSG % dx_check
	ar_cue_delay:
	  	time 1 
		do dummytgt()
	   to fixbreak on +WD0_XY & eyeflag
	   to ar_cue
	ar_cue:
	   do ar_cue_change() 
	   to waitforsac_ARdt on DX_MSG % dx_check
	waitforsac_ARdt:
	   do dx_set_fix(0)
	   to sacon_dots on +WD0_XY & eyeflag
	   to ncerr
	sacon_ARdt:
	   do dx_toggle2(SACMAD, 0, 0, 1000, 3, 1000)
	   rl 50
	   to wait4hit_ARdt on DX_MSG % dx_check
	wait4hit_ARdt:
	   do getsactime()
	   to correct_dots on -WD1_XY & eyeflag
	   to incorrect_dots on -WD2_XY & eyeflag
	   to ncerr


	
	/* OUTCOME STATES
	** NCERR (No-Choice Error)
	**	ERROR
	** CORRECT
	*/


	/* fixation break */
	fixbreak:
		do dummyfpcueoff()
		to fixbreak_score
	fixbreak_score:
		do pr_score_trial(kBrFix,0,1)
		to finish_fixbreak
	finish_fixbreak:
		time 2500	
		do pr_finish_trial()
		rl 0
		to wait2nexttrial
	
	/* no choice */
	ncerr:
		do dummyfpcueoff()
		to ncerr_score
	ncerr_score:
		do pr_score_trial(kNC, 0, 1)
		to finish_ncerr
	finish_ncerr:
		time 1000
		do pr_finish_trial()
		to wait2nexttrial


	/* error */
	error:
		time 1
		do pr_score_trial(kError, 0, 1)
		to finish

	/* pref -- reward! */
	correct:
		time 500
		to correctCD
	correctCD:
		do correctfun()
		to finish 


	correctASL:
		do pr_score_trial(kCorrect, 0, 1)
		to rewardASL
	rewardASL:
		do pr_set_reward(1,100,1,1,100,1)		
		to finish on 0 % pr_beep_reward 

	finish:
		do pr_finish_trial()
		rl 0
		to wait2nexttrial
	wait2nexttrial:
		do dummyfpcueoff()
		to loop
		
abort list:		
	finish
}


/* set to check for fixation bread during task...
** use set_eye_flag to set gl_eye_state to non-zero
** to enable loop
*/
eye_set {
status ON
   begin efirst:
      to etest
   etest:
      to echk on 1 % dx_check_fix
   echk:
      to efail on +WD0_XY & eyeflag
      to etest on 0 % dx_check_fix
   efail:
		do dummyfpcueoff()
		to efail_score
	efail_score:
      to etest

abort list:
}
