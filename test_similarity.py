import numpy as np
vector_a = np.array([1,2,3]) 
vector_b = np.array([1,2,3]) 
dot_product = np.dot(vector_a,vector_b)
magnitude_a = np.linalg.norm(vector_a)
magnitude_b = np.linalg.norm(vector_b)
similarity = dot_product/(magnitude_a*magnitude_b)
print("Similarity :",similarity)
