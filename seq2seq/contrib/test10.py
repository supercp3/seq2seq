import sys

def solve(data):
	len1=len(data)
	len2=len(data[0])
	num=0
	for i in range(len1):
		if data[i][1]==1:
			num+=1
		else:
			if data[i][0]==2:
				num+=1
	return 0,num

x=int(sys.stdin.readline().strip())
data=[]
for i in range(x):
	data.append(list(map(int,sys.stdin.readline().strip().split())))
res=solve(data)
print(res[0],res[1])

