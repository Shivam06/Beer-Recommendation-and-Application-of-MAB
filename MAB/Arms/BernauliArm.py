import numpy as np 

class BernauliArm:
	def __init__(self, p):
		self.p = p
 
	def draw(self):
		return int(np.random.random() < self.p)

if __name__ == "__main__":
	arm = BernauliArm(0.5)
	for i in range(5):
		print arm.draw()

