




proc import datafile= "D:\Chengbinhu\Desktop\Dataset_USF_JHU.csv"out=mydata dbms=csv replace;
run;
proc contents data=mydata;
run;
proc ttest data = Mydata;
class Condition_0_1_;
var Email_1_Time Email_2_Time Email_3_Time Email_4_Time Email_5_Time Email_6_Time Email_7_Time Email_8_Time Email_9_Time Email_10_Time Email_11_Time Email_12_Time Email_13_Time Email_14_Time Email_15_Time Email_16_Time Email_17_Time Email_18_Time Email_19_Time Email_20_Time;
Run;

Data JHUtotal1(DROP = i);
set Mydata;
array emailtime {20} Email_1_Time Email_2_Time Email_3_Time Email_4_Time Email_5_Time Email_6_Time Email_7_Time Email_8_Time Email_9_Time Email_10_Time Email_11_Time Email_12_Time Email_13_Time Email_14_Time Email_15_Time Email_16_Time Email_17_Time Email_18_Time Email_19_Time Email_20_Time;
total = 0;
DO i = 1 to 20;
	total = emailtime[i] + total;
end;
Run;

proc ttest data = jhutotal1;
class Condition_0_1_;
var total;
Run;

proc import datafile= "D:\Chengbinhu\Desktop\Dataset_USF_JHU.csv"out=JHUoveralldata dbms=csv replace;
run;



proc corr data = JHUoveralldata;
var Total_Score_60 Intervention Condition_0_1_ age gender Native_Spk Edu1 Edu2 Habit1 Habit2 SelfEff1 SelfEff2 SelfEff3 SelfEff4 Susp1 Susp2 Susp3 HP1 HP2 HP3 SP1 SP2 SP3 RB1 RB2 HabStgth1 HabStgth2 HabStgth3 HabStgth4 HabStgth5 HabStgth6 HabStgth7 Study_Know;
run;

Proc reg data =  JHUoveralldata;
model Total_Score_60 = SP1;
run;
Proc reg data =  JHUoveralldata;
model Total_Score_60 = SP2;
run;

  proc sgplot data=JHUoveralldata;
	reg x=SP2 y=Total_Score_60 / lineattrs=(color=red thickness=2);
  Title "Regression Line for SP2  to Total performance";
  run;



proc logistic data=Jhuoveralldata desc; 

  model R1_P1_Score = Intervention R1_P1_Time Condition_0_1_ age gender Native_Spk Edu1 Edu2 Habit1 Habit2 / link = glogic selection = forward;

  /*output out = prob PREDPROBS=I; */

run;


proc reg data=Jhuoveralldata; 

  model R1_P1_Score = Intervention R1_P1_Time Condition_0_1_ age gender Native_Spk Edu1 Edu2 Habit1 Habit2 / selection = forward;

  /*output out = prob PREDPROBS=I; */

run;


proc reg data=Jhuoveralldata; 

  model R1_total_Score = Intervention R1_P1_Time R1_P2_Time R1_P3_Time Condition_0_1_ age gender Native_Spk Edu1 Edu2 Habit1 Habit2 / selection = forward SLENTRY= 0.1;

  /*output out = prob PREDPROBS=I; */

run;






proc reg data=Jhuoveralldata; 

  model R1_total_Score = Intervention R1_P1_Time R1_P2_Time R1_P3_Time Condition_0_1_ age gender Native_Spk Edu1 Edu2 Habit1 Habit2 / selection = backward SLSTAY= 0.2;

  /*output out = prob PREDPROBS=I; */

run;


/*Step wise*/

proc reg data=Jhuoveralldata; 

  model Total_Score_60 = Intervention Condition_0_1_ age gender Native_Spk Edu1 Edu2 Habit1 Habit2 SelfEff1 SelfEff2 SelfEff3 SelfEff4 Susp1 Susp2 Susp3 HP1 HP2 HP3 SP1 SP2 SP3 RB1 RB2 HabStgth1 HabStgth2 HabStgth3 HabStgth4 HabStgth5 HabStgth6 HabStgth7 Study_Know / selection = stepwise SLENTRY = 0.5 SLSTAY= 0.2;

  /*output out = prob PREDPROBS=I; */

run;


/*Forward*/

proc reg data=Jhuoveralldata; 

  model Total_Score_60 = Intervention Condition_0_1_ age gender Native_Spk Edu1 Edu2 Habit1 Habit2 SelfEff1 SelfEff2 SelfEff3 SelfEff4 Susp1 Susp2 Susp3 HP1 HP2 HP3 SP1 SP2 SP3 RB1 RB2 HabStgth1 HabStgth2 HabStgth3 HabStgth4 HabStgth5 HabStgth6 HabStgth7 Study_Know / selection = forward SLENTRY= 0.5;

  /*output out = prob PREDPROBS=I; */

run;
/*Backward*/

proc reg data=Jhuoveralldata; 

  model Total_Score_60 = Intervention Condition_0_1_ age gender Native_Spk Edu1 Edu2 Habit1 Habit2 SelfEff1 SelfEff2 SelfEff3 SelfEff4 Susp1 Susp2 Susp3 HP1 HP2 HP3 SP1 SP2 SP3 RB1 RB2 HabStgth1 HabStgth2 HabStgth3 HabStgth4 HabStgth5 HabStgth6 HabStgth7 Study_Know / selection = forward SLENTRY= 0.1;

  /*output out = prob PREDPROBS=I; */

run;

proc logistic data=Jhuoveralldata; 

  model Total_Score_60 = Intervention Condition_0_1_ age gender Native_Spk Edu1 Edu2 Habit1 Habit2 SelfEff1 SelfEff2 SelfEff3 SelfEff4 Susp1 Susp2 Susp3 HP1 HP2 HP3 SP1 SP2 SP3 RB1 RB2 HabStgth1 HabStgth2 HabStgth3 HabStgth4 HabStgth5 HabStgth6 HabStgth7 Study_Know / selection = stepwise SLENTRY = 0.5 SLSTAY= 0.2;

  /*output out = prob PREDPROBS=I; */

run;

PROC UNIVARIATE DATA = Jhuoveralldata PLOT;

var  Total_Score_60;
RUN;

Proc sgplot DATA = Jhuoveralldata;
Title "Distrubution of Time";
VBOX R1_P1_Time R1_P2_Time R1_P3_Time R1_NR_Time Time_R1 R2_P1_Time R2_P2_Time R2_P3_Time R2_Nr_Time Time_R2 R3_P1_Time R3_P2_Time R3_P3_Time R3_Nr_Time Time_R3
RUN;



proc corr data = JHUoveralldata;
var Total_Score_60 R1_Phis_Time R2_Phis_Time R3_Phis_Time R1_P1_Time R1_P2_Time R1_P3_Time R1_NR_Time Time_R1 R2_P1_Time R2_P2_Time R2_P3_Time R2_Nr_Time Time_R2 R3_P1_Time R3_P2_Time R3_P3_Time R3_Nr_Time Time_R3;
run;

proc reg data=Jhuoveralldata; 

  model Total_Score_60 = R1_Phis_Time R2_Phis_Time R3_Phis_Time R1_P1_Time R1_P2_Time R1_P3_Time R1_NR_Time Time_R1 R2_P1_Time R2_P2_Time R2_P3_Time R2_Nr_Time Time_R2 R3_P1_Time R3_P2_Time R3_P3_Time R3_Nr_Time Time_R3/ selection = forward;

  /*output out = prob PREDPROBS=I; */

run;

proc reg data=Jhuoveralldata; 

  model Total_Score_60 = R1_Phis_Time R2_Phis_Time R3_Phis_Time R1_P1_Time R1_P2_Time R1_P3_Time R1_NR_Time Time_R1 R2_P1_Time R2_P2_Time R2_P3_Time R2_Nr_Time Time_R2 R3_P1_Time R3_P2_Time R3_P3_Time R3_Nr_Time Time_R3/ selection = stepwise;

  /*output out = prob PREDPROBS=I; */

run;

proc reg data=Jhuoveralldata; 

  model Total_Score_60 =  R1_P1_Time R1_P2_Time R1_P3_Time R1_NR_Time 
Time_R1 R2_P1_Time R2_P2_Time R2_P3_Time R2_Nr_Time Time_R2 R3_P1_Time R3_P2_Time R3_P3_Time R3_Nr_Time Time_R3/ selection = backward;

  /*output out = prob PREDPROBS=I; */

run;
proc import datafile= "D:\Chengbin\Desktop\matlabplot\phishtimebox.csv"out=timeanova dbms=tab  replace;
getnames=no;
run;

 proc ANOVA data=timeanova;
	
	class VAR2;
	model VAR1 = VAR2;
	means method /hovtest welch;
	run;



PROC GLM DATA = timeanova;
class VAR2;
MODEL VAR1 = VAR2;
LSMEANS VAR2/ PDIFF adjust = TUKEY;
RUN;
QUIT;


/*whole model*/
proc reg data=Jhuoveralldata; 

  model Total_Score_60 = Intervention Condition_0_1_ age gender Native_Spk Edu1 Edu2 Habit1 Habit2 SelfEff1 SelfEff2 SelfEff3 SelfEff4 Susp1 Susp2 Susp3 HP1 HP2 HP3 SP1 SP2 SP3 RB1 RB2 HabStgth1 HabStgth2 HabStgth3 HabStgth4 HabStgth5 HabStgth6 HabStgth7 Study_Know R1_Phis_Time R2_Phis_Time R3_Phis_Time R1_P1_Time R1_P2_Time R1_P3_Time R1_NR_Time Time_R1 R2_P1_Time R2_P2_Time R2_P3_Time R2_Nr_Time Time_R2 R3_P1_Time R3_P2_Time R3_P3_Time R3_Nr_Time Time_R3/ selection = stepwise SLENTRY = 0.5 SLSTAY= 0.2;

  /*output out = prob PREDPROBS=I; */

run;

proc reg data=Jhuoveralldata; 

  model Total_Score_60 = Intervention Condition_0_1_ age gender Native_Spk Edu1 Edu2 Habit1 Habit2 SelfEff1 SelfEff2 SelfEff3 SelfEff4 Susp1 Susp2 Susp3 HP1 HP2 HP3 SP1 SP2 SP3 RB1 RB2 HabStgth1 HabStgth2 HabStgth3 HabStgth4 HabStgth5 HabStgth6 HabStgth7 Study_Know R1_Phis_Time R2_Phis_Time R3_Phis_Time R1_P1_Time R1_P2_Time R1_P3_Time R1_NR_Time Time_R1 R2_P1_Time R2_P2_Time R2_P3_Time R2_Nr_Time Time_R2 R3_P1_Time R3_P2_Time R3_P3_Time R3_Nr_Time Time_R3/ selection = forward SLENTRY= 0.2;

  /*output out = prob PREDPROBS=I; */

run;

proc reg data=Jhuoveralldata; 

  model Total_Score_60 = Intervention Condition_0_1_ age gender Native_Spk Edu1 Edu2 Habit1 Habit2 SelfEff1 SelfEff2 SelfEff3 SelfEff4 Susp1 Susp2 Susp3 HP1 HP2 HP3 SP1 SP2 SP3 RB1 RB2 HabStgth1 HabStgth2 HabStgth3 HabStgth4 HabStgth5 HabStgth6 HabStgth7 Study_Know R1_Phis_Time R2_Phis_Time R3_Phis_Time R1_P1_Time R1_P2_Time R1_P3_Time R1_NR_Time Time_R1 R2_P1_Time R2_P2_Time R2_P3_Time R2_Nr_Time Time_R2 R3_P1_Time R3_P2_Time R3_P3_Time R3_Nr_Time Time_R3/ selection = backward SLSTAY= 0.5;

  /*output out = prob PREDPROBS=I; */

run;
