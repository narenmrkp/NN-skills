Numpy-Nerchuko:
    # videos-(1,2,3)
lst = [1,2,3,4,5] --> np.array(lst)
matrix = [[1,2,3],[4,5,6],[7,8,9]] --> np.array(matrix)
    
np.arange(0,16,3)
np.arange(10,5,-1)
np.zeros(3) --> np.zeros(2,3)
np.ones(5) --> np.ones(3,3)
np.linspace(0,10,3) --> np.linspace(0,10,30)
np.eye(4)
np.random.rand(2) --> np.random.rand(3,3)
np.random.randn(2) --> np.random.randn(3,3)
np.random.randint(1,100) --> np.random.randint(1,100,10)
arr = np.arange(20) --> arr.reshape(4,5)
arr2 = np.random.randint(0,50,10) --> arr2.max() --> arr2.argmax() --> arr2.min() --> arr2.argmin()
    
arr = np.arange(0,11) --> arr[0] --> arr[3:6] --> arr[::-1] --> arr[0:4] = 49
arr = np.arange(0,11) --> slice_arr = arr[0:5] --> slice_arr[:] = 100
arr_copy = arr.copy()
matrix = np.array( [[1,2,3],[4,5,6],[7,8,9]]) --> matrix[0][0] --> matrix[2][2]
matrix.shape
matrix[:2, 1:] --> matrix[1:, 1:]
arr = np.arange(1,15) --> bool_arr = arr > 5 --> arr[bool_arr] --> arr[arr>2]
arr = np.arange(0,10) --> arr + arr --> arr*arr --> arr - arr --> arr/arr
arr**2 --> np.sqrt(arr) --> np.exp(arr) --> np.max(arr) --> arr.max() --> np.sin(arr)
arr = np.random.randint(0,25,6) --> arr.mean() --> arr.min() --> arr.max() --> arr.var() --> arr.std() # std = sqrt(variance)
np.median(arr)
