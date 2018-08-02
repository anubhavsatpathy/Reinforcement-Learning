import collections

dq = collections.deque([1,2,3])
dq.append(4)
print(dq)
dq.popleft()
print(dq)
print(dq[-1])