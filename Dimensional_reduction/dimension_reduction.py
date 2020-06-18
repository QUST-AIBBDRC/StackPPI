import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import KernelPCA,PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import MDS


def zscore_scaler(data):
    data=scale(data)
    return data
def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)#axis represents the obtained mean value by column
    stdVal=np.std(dataMat,axis=0)
    newData=dataMat-meanVal
    new_data=newData/stdVal
    return new_data,meanVal
def covArray(dataMat):
    #obtain the  covariance matrix
    covMat=np.cov(dataMat,rowvar=0)
    return covMat
def featureMatrix(dataMat):
    eigVals,eigVects=np.linalg.eig(np.mat(dataMat))
	#datermine the eigenvalue and eigenvector
    return eigVals,eigVects
def percentage2n(eigVals,percentage=0.99):  
    #percentage represents the rate of contribution
    sortArray=np.sort(eigVals)   #ascending sort 
    sortArray=sortArray[-1::-1]  #descending order
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num #num is the number of remaining principal component
#using kernel principal component to reduce dimension
def KPCA(data,percentage=0.9):
    dataMat = np.array(data)  
    newData=zscore_scaler(data)  
    covMat=covArray(newData)  
    eigVals,eigVects=featureMatrix(covMat)
    n_components = percentage2n(eigVals,percentage)
    #n_component=n_components
    clf=KernelPCA(n_components=n_components,kernel='rbf',gamma=1/1318)#rbf linear poly
    new_x=clf.fit_transform(dataMat)
    return new_x
def LLE(data,n_components=300):
    embedding = LocallyLinearEmbedding(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    return X_transformed
def TSVD(data,n_components=300):
    svd = TruncatedSVD(n_components=n_components)
    new_data=svd.fit_transform(data)  
    return new_data
def mds(data,n_components=300):
    embedding=MDS(n_components=n_components)
    new_data=embedding.fit_transform(data)
    return new_data
