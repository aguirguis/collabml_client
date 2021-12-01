bw='UNLIMITED'	#100
type='sweepExp' #'parallelBaseline'
filename = '{}AllModels_bw{}GPUusagelog'.format(type,bw)
f = open(filename,'r')
lines = f.readlines()
parallel_posts = 4
all_occ = []
all_gpu_mem = []
counter=0
cur_post_time = 0
#making "the next variable" False will measure only the time of receiving replies from Swift (used to calculate comm. time)
measuring_all = True
gpu_mem = False
temp_mem=[]
for line in lines:
#GPU memory printing
  if gpu_mem:
    if line.startswith("GPU memory for"):
      all_gpu_mem.append(float(line.split()[-1]))
      if len(all_gpu_mem) == 12:
#        print(max(all_gpu_mem))
        temp_mem.append(max(all_gpu_mem))
        all_gpu_mem = []
      if len(temp_mem) == 5:				#for now, we run in 5 main batches, each one is of 10k
        print(sum(temp_mem[1:]) / 4)			#the first value is always outlier
        temp_mem=[]
    continue
#The whole process time printing
  if measuring_all:
    if line.startswith('The whole'):
      print(line.split()[-2])
    continue
#The time of receiving results from Swift printing
  if line.startswith('Executing one post'):
    all_occ.append(float(line.split()[-2]))
    if len(all_occ) == parallel_posts:
       cur_post_time+=max(all_occ)
       counter+=1
       all_occ=[]
    if counter == 4:
       print(cur_post_time)
       cur_post_time = 0
       counter=0
