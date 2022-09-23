import numpy as np

# #reshape
# a=np.array([
#     [[1,2,3],[4,5,6]],
#     [[7,8,9],[10,11,12]]
# ])
# b=a.reshape(1,-1)
# print(b)
#
# #flat,遍历每一个元素，for遍历最外层[]内的项
# #区别https://blog.csdn.net/tcy23456/article/details/84362357
# a=np.arange(9).reshape(3,3)
# for row in a:
#     print(row)
# for ele in a.flat:
#     print(ele,end=",")

# #nditer()迭代器
# a=np.array([
#     [[1,2,3],[4,5,6]],
#     [[7,8,9],[10,11,12]]
# ])
# b=a.reshape(1,-1)
# for x in np.nditer(a,flags=['buffered'],op_dtypes=['S']):
#     print(x)
# print(a)
#
# a=np.array([
#     [[1,2,3],[4,5,6]],
#     [[7,8,9],[10,11,12]]
# ])
# b=a.reshape(1,-1)
# a=np.arange(0,60,5).reshape(3,4)
# for x in np.nditer(a.T.copy(order="C")):
#     print(x,end=",")
# print(a)
#

# # #resize
# # reshape只能按照数组原有的维度进行重组不能越界，而resize函数可以越界
# #有返回值np.resize
# a=np.array([
#     [[1,2,3],[4,5,6]],
#     [[7,8,9],[10,11,12]]
# ])
# b=a.reshape(1,-1)
# c=np.resize(a,(4,4))
# print(c)
# print(b)
# print(a)
#
# x=np.arange(12)
# print(x)
#
#
# #无返回值array.resize
# c=x.resize(4,4)
# print(x)
# print(c)
#
# #np.append(arr,v,axis)
# #arr,v维度相同
# a=np.arange(1,7).reshape(2,3)
# # b=[np.arange(7,10).reshape(1,3)]
# # print(b)
# # print(np.arange(7,10))
# #axis=0,保证arr，v列数相同
# print(np.append(a,[np.arange(7,10)],axis=0))
# #axis=1保证arr，v行数相同
# print(np.append(a,[np.zeros(3)+5,np.ones(3)*6+np.arange(1,4)],axis=1))
#
# #np.fromiter(iterable,dtype,count=-1)
# #iterabler可迭代对象，
# list=range(6)
# i=iter(list)
# ff=np.fromiter(i,int)
# print(ff,ff.dtype)
#
# iterable=(x*x for x in range(6))
# a=np.fromiter(iterable,float)
# print(a)
# #
# def gen():
#     for i in range(10):
#         yield i
# a=np.fromiter(gen(),dtype=float,count=3)
# print(a)

# # for x in i:
# #     print(x)
# # print(next(i))


# print(1)
# for y in iter(list):
#     print(y)
#
# #np.dtype([('score','i1')])
# dt=np.dtype([('score','i1')])
# a=np.array([(55,),(75,),(85,)],dtype=dt)
# print(a,a.dtype,a["score"])
#
# #insert
# a=np.array([[1,2],[3,4],[5,6]])
# #默认展开为一维,
#
# print(np.insert(a,3,[11,12]))
# print(a)
# print(np.insert(a,1,[11],axis=0))
#
# print(np.insert(a,1,[12],axis=1))
#
# #广播
# a=np.array([[0,0,0],
#             [10,10,10],
#             [20,20,20],
#             [30,30,30]])
# b=np.array([1,2,3])
# #扩展为 [[1,2,3],
# #       [1,2,3],
# #       [1,2,3],
# #       [1,2,3]]
# print(a+b)

# #np.ravel()
# #一维展开,默认order="C",即行展开，order="F"为列展开
# a=np.arange(8).reshape(2,4)
# print(a)
# print(a.ravel())
# print(a.ravel(order='F'))

# #转置
# #np.transpose(arr,axe),axe二维索引(0,1),三维(0,1,2),反转即转置axe可省略,与a.T一样
# a=np.arange(12).reshape(3,4)
# print(a)
# print(np.transpose(a))

# #numpy.swapaxes()将两个维度(轴)调换
# a=np.arange(27).reshape(3,3,3)
# print(a)
# print(np.swapaxes(a,2,0))
# # 轴移动到特定位置
# print(np.rollaxis(a,1))

# #np.broadcast()
# a=np.array([[1],[2],[3]])
# print(a.shape)
# b=np.array([4,5,6])
# print(b.shape)
# d=np.broadcast(a,b)#d的元素是一个元组
# print(d.shape)
#
# r,c=d.iters
# print(next(r),next(c))
# print(next(r),next(c))
# print(next(r),next(c))
# print(next(r),next(c))
# print(next(r),next(c))
# print(next(r),next(c))
# print(next(r),next(c))
# print(next(r),next(c))
# print(next(r),next(c))
#
# e=np.broadcast(a,b)
#
# f=np.empty(e.shape)
# f.flat=[x+y for (x,y) in e]
# print(f)
# print(a+b)

# #broadcast_to()变成指定形状
# a=np.arange(4).reshape(1,4)
# print("原数组",a)
# print(np.broadcast_to(a,(4,4)))

# #np.expand_dims()插入新轴
# x=np.arange(1,5).reshape(2,2)
# print('x',x)
# y=np.expand_dims(x,axis=2)
# print(y)
# print(x.shape,y.shape)

# #numpy.squeeze(arr,size)删除维度值为一的轴
# x=np.arange(3).reshape(1,3,1)
# print(x.shape,x)
# y=np.squeeze(x)
# print(y,y.shape)
# y=np.squeeze(x,axis=(0,2))#axis=0/axis=2
# print(y,y.shape)

# #np.concatenate(（）,axis=0)#沿指定轴连接形状相同的数组
# a=np.arange(10,41,10).reshape(2,2)
# b=np.arange(50,90,10).reshape(2,2)
# c=np.concatenate((a,b),axis=1)
# d=np.concatenate((a,b))
# print(c,'c.shape=',c.shape)
# print(d,'d.shape=',d.shape)

# #np.vstack()#垂直堆叠
# a=np.arange(1,5).reshape(2,2)
# b=np.arange(5,9).reshape(2,2)
# c=np.vstack((a,b))
# print(c)
#
# #np.hstack()
# #hstack、vstack，stack
# #https://blog.csdn.net/csdn15698845876/article/details/73380803
# a=np.arange(1,5).reshape(2,2)
# b=np.arange(5,9).reshape(2,2)
# c=np.hstack((a,b))
# print(c)

# #numpy.split(arr,a,axis)#a为整数或数组，整数为平均分割，数组为沿轴分割
# a=np.arange(6)
# print(a)
# b=np.split(a,2)
# print('b',b)
# b=np.split(a,[3,4])
# print('b',b)

# #stack(arr,axis)axis=0增加行数，axis=1竖着堆叠，增加列数
# a=[[1,2,3,4],
#    [5,6,7,8],
#    [9,10,11,12]]
# print("列表a如下：")
# print(a)
#
# print("增加一维，新维度的下标为0")
# c=np.stack(a,axis=0)
# print(c,c.shape)
#
# print("增加一维，新维度的下标为1")
# c=np.stack(a,axis=1)
# print(c,c.shape)

# # append(arr,values,axis=None),axis=None默认一维
# # insert(arr,obj,values,axis),
# # delete(arr,obj,axis)
# a=np.arange(6).reshape(2,3)
# print(a)
# b=np.append(a,[[7,8,9]],axis=0)
# print(b)
# c=np.insert(b,3,[10,11,12],axis=0)
# print(c)
# d=np.insert(c,3,6,axis=1)
# print(d)

# # argwhere()#返回非零元素的索引
# x=np.arange(6).reshape(2,3)
# print(np.argwhere(x))#1,2,3,4,5
# print(np.argwhere(x>3))#4,5

# #np.unique(arr,return_index,return_inverse,return_counts)
# a=np.array([1,1,2,2,3,3,4,4,5,1,2,4,6,5])
# u=np.unique(a)
# print("unique",u)
# u,i=np.unique(a,return_index=True)#原数组中的位置
# print("unique,returnindex",u,i)
# u,i=np.unique(a,return_inverse=True)#新数组中的位置
# print("unique,return_counts",u,i)
# u,i=np.unique(a,return_counts=True)#去重后元素在原数组中的位置
# print("unique,return_counts",u,i)

# #位
# a=10
# b=12
# #bitwise_and()位与
# print("a's bit",bin(a))
# print("b's bit",bin(b))
# print("a bit_wise b",np.bitwise_and(a,b))
# #bitwise_or()位或
# orb=np.bitwise_or(a,b)
# print(orb)
# #invert() 取反,np.binary_repr 函数用来设置二进制数的位数
# print(np.binary_repr(a,8))
# a=np.invert(a)
# print(np.binary_repr(a,8))
# #left_shift()位移，right_shift()位移
#
# #np.around()四舍五入
# aa=np.random.rand(5)
# print(aa)
# print(np.around(aa))
# print(np.floor(aa))
# print(np.ceil(aa))


# #np.empty(shape,dtype=float,order='C')
# #np.zeros(shape,dtype=float,order='C')
# #np.ones(shape,dtype=None,order='C')
# #np.asarray(sequence,dtype=None,order=None)
# arr=np.empty((3,3),dtype=int)
# print(arr)
#
# arr1=np.zeros((3,3))
# print(arr1)
#
# arr2=np.ones((3,3))
# print(arr2)
#
# arr3=np.asarray([1,2,3,4,5])
# print(type(arr3),arr3)

# #+-*/
# a=np.arange(9,dtype=np.float_).reshape(3,3)
# print(a)
# b=np.array([4,4,4])
# print(b)
#
# print(np.add(a,b))
# print(np.subtract(a,b))
# print(np.multiply(a,b))
# print(np.divide(a,b))

# #np.reciprocal取倒数,int类型结果为0
# a=np.arange(1,6,dtype=np.float_)
# print(a)
# b=np.reciprocal(a)
# print(b)
# print(np.reciprocal(2))

# #np.power(a,b) b作a的幂==a**b
# a=np.array([10,100,1000])
# print(a)
# print(np.power(a,2))
# print(a**2)#等价
# b=np.arange(1,4)
# print(np.power(a,b))

# #np.mod()==np.remainder()取余和取模
# a=np.array([11,22,33])
# b=np.array([3,5,7])
# print(np.mod(a,b))
#
# print(np.remainder(a,b))

# #np.real(a)返回实部
# #np.imag(a)返回虚部
# #np.conj(a)返回共轭
# #np.angel(a，deg=False)返回参数角度，deg=True为角度制
# a=np.array([5.1j,1+2.0j,2j,3+4j])
# print(np.real(a))
# print(np.imag(a))
# print(np.conj(a))
# print(np.angle(a))
# print(np.angle(a,deg=True))

# #np.amin(arr,axis),np.amax(arr,axis)
# #np.ptp(arr,axis)最大差值
# a=np.array([[2,3,4],[4,6,5],[0,1,3]])
# print(a)
# print(np.amin(a))
# print("每行最小(axis=1)",np.amin(a,1))
# print("每列最小(axis=0)",np.amin(a,0))
#
# print(np.amax(a))
# print("每行最大(axis=1)",np.amax(a,1))
# print("每列最大(axis=0)",np.amax(a,0))
#
# print(np.ptp(a))
# print(np.ptp(a,0))
# print(np.ptp(a,1))

# #百分位数
# #np.percentile(a,q,axis,)
# #q为百分数位，输出结果在数组中为q%
# a=np.random.randint(100,size=9).reshape(3,3)
# print(a)
# print(np.percentile(a,50))
# print(np.percentile(a,50,1))
# print(np.percentile(a,50,0))

# #平均数np.mean(arr,axis)
# a=np.arange(1,10).reshape(3,3)
# print(a)
# print(np.mean(a))
# print(np.mean(a,0))
# print(np.mean(a,1))

# #加权平均np.average(arr,axis,weights,returned)
# a=np.arange(6).reshape(3,2)
# b=np.random.randint(100,size=6).reshape(3,2)
# print(a,b)
# print(np.average(a,weights=b))
# print(np.average(a,axis=1,weights=b[1,:]))#weights=b也可以

# #np.var()方差
# print(np.var([1,2,3,4]))
# #np.std()标准差
# print(np.std([1,2,3,4]))

# #np.dot()点积,按照矩阵乘法规则
# A=[1,2,3]
# B=[4,5,6]
# print(np.dot(A,B))
# a=np.arange(1,5).reshape(2,2)
# b=np.arange(5,9).reshape(2,2)
# dot=np.dot(a,b)
# print(dot)

# #np.vdot()内积，向量点积结果
# a=np.array([[100,200],[23,12]])
# b=np.array([[10,20],[12,21]])
# print(np.vdot(a,b))

# #inner() 函数的计算过程是 A 数组的每一行与 B 数组的每一行相乘再相加
# #一维数组时与dot相同
# #内积
# #A= a  b
# #   c  d
# #B= e  f
# #   g  h
# #np.dot(A,B)=ae+bg,af+bh
# #            ce+bg,df+dh
# #np.inner(A,B)=ae+bf,ag+bh
# #              ce+df,cg+dh
# A=[[1,10],[100,1000]]
# B=[[1,2],[3,4]]
# print(np.inner(A,B))
# print(np.dot(A,B))

# #np.matmul()低维时和dot（）一样
# a=np.arange(1,10).reshape(3,3)
# b=np.arange(11,20).reshape(3,3)
# print(np.matmul(a,b))

# #np.linalg.det(arr)行列式
# a = np.array([[1,2],[3,4]])
# print(np.linalg.det(a))

# #np.linalg.solve()求解线性矩阵
# # 3X  +  2 Y + Z =  10
# # X + Y + Z = 6
# # X + 2Y - Z = 2
# m=np.array([[3,2,1],
#            [1,1,1],
#             [1,2,-1]])
# n=np.array([10,6,2])
# x=np.linalg.solve(m,n)
# print(x)

# #numpy.linalg.inv()求逆
# a=np.arange(1,5).reshape(2,2)
# print(np.linalg.inv(a))


