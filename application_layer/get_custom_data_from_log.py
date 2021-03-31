bw='UNLIMITED'
f = open('ExpToBenchmarkALLDataMovementCost_bw{}_withParallelPosts_shot2'.format(bw),'r')
lines = f.readlines()
parallel_posts = 2
all_occ = []
for line in lines:
  if line.startswith('The whole'):
    print(line.split()[-2])
#  if line.startswith('Executing one post'):
#    all_occ.append(float(line.split()[-2]))
#    if len(all_occ) == parallel_posts:
#      print(max(all_occ))
#      all_occ=[]
