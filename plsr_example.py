import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def import_dataset(ds_name="octane"):
    '''
    ds_name: Name of the dataset ("octane", "gasoline")
    Returns:
    wls: Numpy ndarray: List of wavelength
    xdata: Pandas DataFrame: Measurements
    ydata: Pandas Series: Octane numbers
    '''

    if ds_name == "octane":
        oct_df = pandas.read_excel("octane.xlsx")
        wls = np.array([ int(i) for i in oct_df.columns.values[2:]])
        xdata = oct_df.loc[:,'1100':]
        ydata = oct_df['Octane number']
    elif ds_name == "gasoline":
        import re

        gas_df = pandas.read_csv("gasoline.csv")
        reg = re.compile('([0-9]+)')
        wls = np.array([ int(reg.findall(i)[0]) for i in gas_df.columns.values[1:]])
        xdata = gas_df.loc[:,'NIR.900 nm':]
        ydata = gas_df['octane']
    else:
        exit("Invalid Dataset")

    return wls, xdata, ydata

if __name__ == "__main__":

    wls, xdata, ydata = import_dataset("gasoline")

    total_variance_in_y = np.var(ydata, axis = 0)
    nc=10

    pls = PLSRegression(n_components=nc)
    pls2 = PLSRegression(n_components=2)
    pca = PCA(n_components=2)
    #pls2 = PLSCanonical(n_components=nc)
    pls.fit(xdata, ydata)
    pls2.fit(xdata, ydata)
    pca.fit(xdata)

    y2fit = pls2.predict(xdata)

    pca_scores = pca.transform(xdata)
    pcr = LinearRegression().fit(pca_scores[:,0:2], ydata)
    yPfit = pcr.predict(pca.transform(xdata))

    variance_in_y = np.var(pls.y_scores_, axis = 0)
    fractions_of_explained_variance = variance_in_y / total_variance_in_y

    TSS = np.sum((ydata-np.mean(ydata))**2 )


    print(np.shape(y2fit[:,0]))
    RSS_PLS = np.sum(np.subtract(ydata,y2fit[:,0])**2 )
    r2PLS = 1 - RSS_PLS/TSS

    RSS_PCR = np.sum((ydata-yPfit)**2 )
    r2PCR = 1 - RSS_PCR/TSS

    r2_sum = 0

    fev = []

    for i in range(0,nc):
        Y_pred=np.dot(pls.x_scores_[:,i].reshape(-1,1),
                        pls.y_loadings_[:,i].reshape(-1,1).T) * ydata.std(axis=0, ddof=1) + ydata.mean(axis=0)
        r2_sum += round(r2_score(ydata,Y_pred),3)
        fev.append(r2_sum)
        print('R^2 for %d component: %g, cummulative: %g' %(i+1,round(r2_score(ydata,Y_pred),3), r2_sum))

    print('R^2 for all components (): %g' %r2_sum) #Sum of above

    f, (ax, ax2) = plt.subplots(1,2)

#    ax.plot([np.sum(fractions_of_explained_variance[0:i]) for i in range(len(fractions_of_explained_variance))],'-bo')
    ax.plot(range(1,11),fev,'-bo')
    ax.set_xlabel("Number of PLS Components")
    ax.set_ylabel("Percent Variance Explained in Y")


    print("asdasd")
    ax2.plot(ydata, y2fit, ' ob')
    ax2.plot(ydata, yPfit, ' ^r')
    ax2.set_xlim(83,90)
    ax2.set_ylim(83,90)
    ax2.set_xlabel("Observed Response")
    ax2.set_ylabel("Fitted Response")
    ax2.legend(["PLSR with 2 Components  (R2 = %2.4f)" % (r2PLS),
                "PCR with 2 Components  (R2 = %2.4f)" % (r2PCR)])


    plt.show()
