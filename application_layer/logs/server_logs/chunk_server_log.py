filename = "logserver_withBatchAdaptation_b200_afterfix"
#B=[1000, 2000,3000,4000, 5000, 6000,7000,8000, 10000, 12000,14000]
B=[2000,4000,6000,8000,10000,12000]
with open(filename) as f:
    lines = f.readlines()
    i = -1
    out_file = open("first_part_withadaptation_afterfix_b200","w")
    for line in lines:
        if "start: 0" in line:
            out_file.close()
            i=i+1
            out_file =open(f"server_withadaptation_b{B[i]}_afterfix_b200", "w")
        out_file.write(line)
