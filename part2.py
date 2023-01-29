from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from part1.py import encoder

def run_part2(image):
    
    if type(image[0]) == "array" and len(image[0]) != 28:
        infile = open("part2.pickle",'rb')
        qsvc = pickle.load(infile)
        infile.close()
        # reduce dimensions
        n_dim = 28
        pca = PCA(n_components=n_dim).fit(image)
        image = pca.transform(image)

        # standardize
        std_scale = StandardScaler().fit(image)
        image = std_scale.transform(image)

        # Normalize
        minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
        image = minmax_scale.transform(image)
    
    encd = encoder(image)
    return qsvc.predict(image)