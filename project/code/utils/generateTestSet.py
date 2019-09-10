import h5py
import os


def main(filePath,outPath):
    os.makedirs(outPath, exist_ok=True)
    os.makedirs(os.path.join(outPath, 'images'))
    # with


if __name__ == '__main__':
    filePath = 'C:\\Git\\instance-seg\\cvppp\\CVPPP2017_testing_images.h5'
    outPath = 'C:\\Git\\instance-seg\\cvppp\\formattedA1Only\\test\\'
    main(filePath, outPath)
