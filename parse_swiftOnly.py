import os
tenants = [2,4,6,8,10]
avg = []
maxi = []
for num_tenants in tenants:
    filename=f"swiftOnlyRes_p{num_tenants}"
    with open(filename,"r") as f:
        lines = f.readlines()
        times = []
        for line in lines:
            if line.startswith("The whole process"):
                times.append(float(line.split()[-2]))
    maxi.append(max(times))
    avg.append(sum(times)/len(times))
print(f"Averages: {avg}")
print(f"Makespan: {maxi}")
