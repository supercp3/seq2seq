import sys


def solve(data,n):
	mark=0
	sum=0
	for i in range(n-1):
		count=0
		for j in range(i+1,n):
			if data[i]>data[j]:
				count++
			sum+=count
			if count-i>max:
				max=count-i
				mark=i
	return sum-max,mark+1
n=int(sys.stdin.readline().strip())
data=list(map(int,sys.stdin.readline().strip().split()))
res=solve(data,n)
print(res)