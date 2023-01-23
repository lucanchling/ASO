import numpy as np
from utils import InitIcp
from utils import RotateMatrix

def main():

    vector = np.array([0.5,0.4,0.8])
    vecotor = vector/np.linalg.norm(vector)
    angle = 1.15

    print(RotateMatrix(vector,angle))

    print(InitIcp.RotationMatrix(InitIcp,vecotor,angle))



if __name__ == "__main__":
    main()