import numpy as np

M_PI=3.14
class KernelParameters():
    def __init__(self):
        self.r=1
        self.g=1
        self.b=1
        self.lamb=1
        self.sigma=0.6
        self.omega=M_PI
        self.phi=0
        self.theta=0

    def generate(self, kernelId, numberOfKernels, kernelSize):
        if ((numberOfKernels == 32) and (kernelSize == 5)):
            self.generateCifarLike(kernelId, kernelSize)
        elif ((numberOfKernels == 64) and (kernelSize == 7)):
            self.generateGoogleNetLike(kernelId, kernelSize)
        elif ((numberOfKernels == 64) and (kernelSize == 3)):
            self.generateVggLike(kernelId, kernelSize)
        elif ((numberOfKernels == 96) and (kernelSize == 11)):
            self.generateAlexNetLike(kernelId, kernelSize)
        else:
            print("No predefined gabor filters for this topology.")

    def generateCifarLike(self, kernelId, kernelSize):
        self.lamb = 0.5

        if (kernelId < 8):
            self.omega = M_PI * (kernelSize - 1) / 2 / 1
            self.theta = (kernelId % 8) * M_PI / 8
        elif (kernelId < 14):
            self.omega = M_PI * (kernelSize - 1) / 2 / 2
            self.theta = ((kernelId - 2) % 6) * M_PI / 6 + M_PI / 12
        elif (kernelId < 16):
            self.lamb = 0.5
            self.sigma = 0.75
            self.omega = M_PI * (kernelSize - 1) / 2 / 8
            self.phi = (kernelId % 2) * M_PI
            self.r = 1
            self.g = -1
            self.b = 1
        else:
            self.omega = M_PI * (kernelSize - 1) / 2 / 4
            self.theta = (kernelId % 4) * M_PI / 2 + M_PI / 4 + M_PI / 8
            self.phi = M_PI / 2

        if (kernelId >= 30):
            self.theta = (kernelId % 2) * M_PI / 2 + M_PI / 4 + M_PI / 8
            self.r = 1
            self.g = 1
            self.b = 0
        elif (kernelId >= 28):
            self.theta = (kernelId % 2) * M_PI / 2 + M_PI / 4 - M_PI / 8
            self.r = -1
            self.g = 1
            self.b = -1
        elif (kernelId >= 24):
            self.r = -1
            self.g = 1
            self.b = 1
        elif (kernelId >= 20):
            self.r = 1
            self.g = 0
            self.b = -1

    def generateGoogleNetLike(self, kernelId, kernelSize):
        if (kernelId < 32):
            rotation = kernelId / 8
            frequency = kernelId % 8
            phase = kernelId % 2
            self.lamb = 1 / (1 + frequency / 8.)
            self.sigma = 0.4 + 0.2 * frequency / 8
            self.omega = M_PI * (kernelSize - 1) / 2 / (1 + frequency / 2.)
            self.phi = phase * M_PI + M_PI * 12 / 32
            self.theta = rotation * M_PI / 4
        elif (kernelId < 40):
            self.sigma = 0.45
            self.lamb = 0.5
            self.omega = M_PI * (kernelSize - 1) / 2
            self.theta = (kernelId % 8) * M_PI / 8
            self.phi = M_PI
        elif (kernelId < 46):
            phase = (kernelId - 1) / 3
            size = (kernelId - 1) % 3
            self.lamb = 1 / (1 + size / 2.)
            self.sigma = 1. / (2.5 - size / 2.)
            self.omega = M_PI / 4
            self.phi = phase * M_PI
            self.r = 0.25
            self.g = -1
            self.b = 1
        elif (kernelId < 48):
            self.lamb = 2. / 3
            self.sigma = 1
            self.omega = M_PI * (kernelSize - 1) / 2 / 12
            self.theta = (kernelId % 8) * M_PI + M_PI / 8
            self.phi = M_PI / 2
            self.r = 1
            self.g = 0.1
            self.b = -0.5
        elif (kernelId < 56):
            self.lamb = 2. / 3
            self.sigma = 1
            self.omega = M_PI * (kernelSize - 1) / 2 / 12
            self.theta = (kernelId % 8) * M_PI / 4 + M_PI / 32
            self.phi = M_PI / 2
            self.r = -0.5
            self.g = 0.1
            self.b = 1
        elif (kernelId < 60):
            self.lamb = 1
            self.sigma = 1
            self.omega = M_PI * (kernelSize - 1) / 2 / 12
            self.theta = (kernelId % 8) * M_PI / 2 + M_PI / 8
            self.phi = M_PI / 2
            self.r = 0.25
            self.g = -1
            self.b = 1
        else:
            self.lamb = 2. / 3
            self.sigma = 1
            self.omega = M_PI * (kernelSize - 1) / 2 / 12
            self.theta = (kernelId % 8) * M_PI / 2 + M_PI / 8
            self.phi = M_PI / 2
            self.r = -1
            self.g = -1
            self.b = 1

    def generateVggLike(self, kernelId, kernelSize):
        self.generateGoogleNetLike(kernelId, kernelSize)
        self.sigma = 1

    def generateAlexNetLike(self, kernelId,    kernelSize):
        self.lamb = 1. / 3

        if (kernelId < 48):
            rotation = kernelId / 8
            frequency = kernelId % 8
            phase = kernelId % 2
            self.lamb /= (1 + frequency / 8.)
            self.sigma = 0.5 + 0.2 * frequency / 8
            self.omega = M_PI * (kernelSize - 1) / 2 / (1 + frequency / 2.)
            self.phi = phase * M_PI + M_PI * 12 / 32
            self.theta = rotation * M_PI / 6
        elif (kernelId < 56):
            phase = kernelId / 4
            size = kernelId % 4
            self.lamb /= (1 + size / 2.)
            self.sigma = 1. / (2.5 - size / 2.)
            self.omega = M_PI / 4
            self.phi = phase * M_PI
            self.r = 0.25
            self.g = -1
            self.b = 1
        elif (kernelId < 60):
            self.lamb /= 1.5
            self.sigma = 0.75
            self.omega = M_PI * (kernelSize - 1) / 2 / 8
            self.theta = (kernelId % 4) * M_PI / 2 + M_PI / 8
            self.phi = M_PI / 2
            self.r = -1
            self.g = 1
            self.b = -0.5
        elif (kernelId < 64):
            self.lamb /= 3
            self.sigma = 2
            self.omega = M_PI * (kernelSize - 1) / 2 / 8
            self.theta = (kernelId % 4) * M_PI / 2 + M_PI / 8
            self.phi = M_PI / 2
            self.r = 1
            self.g = -0.5
            self.b = -0.75
        elif (kernelId < 72):
            self.lamb /= 1.5
            self.sigma = 0.75
            self.omega = M_PI * (kernelSize - 1) / 2 / 4
            self.theta = (kernelId % 8) * M_PI / 4 + M_PI / 32
            self.phi = M_PI / 2
            self.r = 1
            self.g = 0.1
            self.b = -0.75
        elif (kernelId < 80):
            self.lamb /= 1.5
            self.sigma = 1
            self.omega = M_PI * (kernelSize - 1) / 2 / 12
            self.theta = (kernelId % 8) * M_PI / 4 + M_PI / 32
            self.phi = M_PI / 2
            self.r = -0.5
            self.g = 0.1
            self.b = 1
        elif (kernelId < 88):
            self.lamb /= 2.5
            self.sigma = 1
            self.omega = M_PI * (kernelSize - 1) / 2 / 8
            self.theta = (kernelId % 8) * M_PI / 4 + M_PI / 32
            self.phi = M_PI / 2
            self.r = -1
            self.g = -1
            self.b = 1
        elif (kernelId < 92):
            self.omega = M_PI * (kernelSize - 1) / 2 / 16
            self.theta = (kernelId % 4) * M_PI / 2 + M_PI / 16
            self.phi = M_PI / 2
        else:
            self.lamb /= 4
            self.sigma = 0.75
            self.omega = M_PI * (kernelSize - 1) / 2 / 4
            self.theta = (kernelId % 8) * M_PI / 4 + M_PI / 32
            self.phi = M_PI / 2
            self.r = -1
            self.g = -1
            self.b = 1

class KernelGenerator:
    def __init__(self, numberOfKernels,    kernelSize):
        self.numberOfKernels = numberOfKernels
        self.kernelSize = kernelSize
        self.kernels = self.getNumberOfElements()

    def generate(self):
        for kernelId in range(self.numberOfKernels):
            self.generateKernel(kernelId)

    def getKernelData(self):
        return self.kernels

    def getSizeOfKernelData(self):
        return self.getNumberOfElements()

    def getNumberOfElements(self):
        #return self.numberOfKernels * self.kernelSize * self.kernelSize
        return np.zeros((self.numberOfKernels, self.kernelSize, self.kernelSize))

    def generateKernel(self, kernelId):
        param = KernelParameters()
        param.generate(kernelId, self.numberOfKernels, self.kernelSize)

        for ky in range(self.kernelSize):
            for kx in range(self.kernelSize):
                x = 2. * kx / (self.kernelSize - 1) - 1
                y = 2. * ky / (self.kernelSize - 1) - 1

                dis = np.exp(-(x * x + y * y) / (2 * param.sigma * param.sigma))
                arg = x * np.cos(param.theta) - y * np.sin(param.theta)
                per = np.cos(arg * param.omega + param.phi)
                val = param.lamb * dis * per

                self.kernels[kernelId, kx, ky] = val
                # kernels[kx + kernelSize * (ky + kernelSize * (0 + 3 * kernelId))] =
                #         (Dtype)(param.r * val)
                # kernels[kx + kernelSize * (ky + kernelSize * (1 + 3 * kernelId))] =
                #         (Dtype)(param.g * val)
                # kernels[kx + kernelSize * (ky + kernelSize * (2 + 3 * kernelId))] =
                #         (Dtype)(param.b * val)


if __name__ == '__main__':
    kernelGenerator = KernelGenerator(64, 7)
    kernelGenerator.generate()

