import scipy

path= "/home/king/PycharmProjects/DataMerger/Data/youtube_rgb/image&label/1/anotation.mat"
data=scipy.io.loadmat(path)
for k,v in data.items():
    print(k)
    print(v)
    if k=="box":
        print(len(v))

