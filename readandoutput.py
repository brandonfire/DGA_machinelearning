from __future__ import print_function


f = open("Dataset_USF_JHU40.csv","r")

fi = open("JHUoutput3.csv","w")

for line in iter(f):
    score = int(line.split(",")[0])
    print(score)
    if score < 40:
        print("1,0,0" , file = fi)
    elif score >= 40 and score <50:
        print("0,1,0", file = fi)
    elif score >= 50:
        print("0,0,1", file = fi)
#    elif score >=40 and score <45:
#        print("0,0,0,1,0,0", file = fi)
#    elif score >=45 and score <50:
#        print("0,0,0,0,1,0", file = fi)
#    elif score >=50:
#        print("0,0,0,0,0,1", file = fi)
    
f.close()
fi.close()
