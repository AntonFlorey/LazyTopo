import numpy as np

class AffineTransform2D():
    """ Affine Transformation in 2D Space """
    def __init__(self, linear_part = np.eye(2), affine_part = np.zeros(2)):
        self.A = linear_part
        self.t = affine_part

    def inverse(self):
        A_inv = np.linalg.inv(self.A)
        return AffineTransform2D(A_inv, -np.dot(A_inv, self.t))

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return np.dot(self.A, other) + self.t
        else:
            raise NotImplementedError("Cant multiply affine 2D transformation with", other)

    def __matmul__(self, other):
        if isinstance(other, AffineTransform2D):
            return AffineTransform2D(self.A @ other.A, np.dot(self.A, other.t) + self.t)
        else:
            raise NotImplementedError("Cant multiply affine 2D transformation with", other)

    def to_numpy_array(self):
        mat = np.eye(3)
        mat[:2,:2] = self.A
        mat[:2,2] = self.t
        return mat

    def __str__(self):
        return "2D Affine Transformation\nLinear Part:\n" + self.A.__str__() + "\nAffine Part:\n" + self.t.__str__()
