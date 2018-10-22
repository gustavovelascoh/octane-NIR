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
from sklearn.model_selection import cross_validate


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

    dataset = "gasoline"
    wls, xdata, ydata = import_dataset(dataset)
    xdata_norm = xdata-np.mean(xdata)
    ydata_norm = ydata-np.mean(ydata)

    total_variance_in_y = np.var(ydata, axis = 0)
    nc=10

    pls = PLSRegression(n_components=nc)
    pls2 = PLSRegression(n_components=2)
    pls3 = PLSRegression(n_components=3)

    plsC = PLSCanonical(n_components=nc)

    pca = PCA(n_components=nc)
    pca2 = PCA(n_components=2)

    pls.fit(xdata, ydata)
    pls2.fit(xdata, ydata)
    pls3.fit(xdata, ydata)

    plsC.fit(xdata_norm, ydata_norm)

    pca.fit(xdata)
    pca2.fit(xdata)

    yfit = pls.predict(xdata)
    y2fit = pls2.predict(xdata)

    xcfit, ycfit = plsC.transform(xdata_norm, ydata_norm)

    pca_scores = pca.transform(xdata)
    pca2_scores = pca2.transform(xdata)

    pcr = LinearRegression().fit(pca_scores, ydata)
    pcr2 = LinearRegression().fit(pca2_scores, ydata)

    yPfit = pcr.predict(pca.transform(xdata))
    yP2fit = pcr2.predict(pca2.transform(xdata))

    variance_in_y = np.var(pls.y_scores_, axis = 0)
    fractions_of_explained_variance = variance_in_y / total_variance_in_y

    TSS = np.sum((ydata-np.mean(ydata))**2 )
    TSSc = np.sum((ydata_norm-np.mean(ydata_norm))**2 )

    print(np.shape(y2fit[:,0]))
    print(np.shape(ycfit))

    RSS_PLS2 = np.sum(np.subtract(ydata,y2fit[:,0])**2 )
    r2PLS2 = 1 - RSS_PLS2/TSS

    RSS_PCR2 = np.sum((ydata-yP2fit)**2 )
    r2PCR2 = 1 - RSS_PCR2/TSS

    RSS_PLS = np.sum(np.subtract(ydata,yfit[:,0])**2 )
    r2PLS = 1 - RSS_PLS/TSS

    RSS_PCR = np.sum((ydata-yPfit)**2 )
    r2PCR = 1 - RSS_PCR/TSS

    RSS_PLSC = np.sum(np.subtract(ydata_norm,ycfit)**2 )
    r2PLSC = 1 - RSS_PLSC/TSSc

    #print("ycfit: ",ycfit)
    #print("ydata_norm: ",ydata_norm)

    print("r2PLS: %s, r2PLSC: %s" % (r2PLS, r2PLSC))

    r2_sum = 0
    r2_sumc = 0

    fev = []
    fevc = []

    for i in range(0,nc):
        Y_pred=np.dot(pls.x_scores_[:,i].reshape(-1,1),
                        pls.y_loadings_[:,i].reshape(-1,1).T) * ydata.std(axis=0, ddof=1) + ydata.mean(axis=0)
        r2_sum += round(r2_score(ydata,Y_pred),3)
        fev.append(r2_sum)
        print('R^2 for %d component: %g, cummulative: %g' %(i+1,round(r2_score(ydata,Y_pred),3), r2_sum))

        Y_predc=np.dot(plsC.x_scores_[:,i].reshape(-1,1),
                        plsC.y_loadings_[:,i].reshape(-1,1).T)
        #print("YPRED: ", Y_predc)
        Y_predc=Y_predc - np.mean(Y_predc)
        #print("YPRED: ", Y_predc)
        r2_sumc += round(r2_score(ydata_norm,Y_predc),3)
        fevc.append(r2_sumc)

    print('R^2 for all components (): %g' %r2_sum) #Sum of above

    # for i in range(0,nc):
    #     Y_pred=np.dot(pls.x_scores_[:,i].reshape(-1,1),
    #                     pls.y_loadings_[:,i].reshape(-1,1).T) * ydata.std(axis=0, ddof=1) + ydata.mean(axis=0)
    #     r2_sum += round(r2_score(ydata,Y_pred),3)
    #     fev.append(r2_sum)
    #     print('R^2 for %d component: %g, cummulative: %g' %(i+1,round(r2_score(ydata,Y_pred),3), r2_sum))
    #
    # print('R^2 for all components (): %g' %r2_sum) #Sum of above

    #asdasd
    total_variance_in_x = np.sum(np.var(xdata, axis = 0))

    print("shape: TotalXvar ", np.shape(total_variance_in_x))
    print("shape: Xscores ", np.shape(pls.x_scores_))

    # variance in transformed X data for each latent vector:
    variance_in_x = []
    for i in range(0,nc):
        variance_in_x.append(
                        np.var(
                            np.dot(
                                pls.x_scores_[:,i].reshape(-1,1),
                                pls.x_loadings_[:,i].reshape(-1,1).T)
                            ,axis = 0)
                            )

    print("shape: Xvar ", np.shape(variance_in_x))

    # normalize variance by total variance:
    fractions_of_explained_variance = np.sum(variance_in_x / total_variance_in_x, axis=1)

    # for i in range(1,11):
    #     x_partial = pls.x_scores_[:,0:i] * pls.x_loadings_[0:i,:]

    # CROSS VALIDATION STAGE
    plscv_err = []
    pcrcv_err = []

    xdata_norm = xdata-np.mean(xdata)

    print(np.shape(pca.singular_values_))
    print(np.shape(pca.components_))


    for i in range(1,nc+1):
        pls_cv = PLSRegression(n_components=i)
        plsrcv  = cross_validate(pls_cv, xdata, ydata, cv=10, scoring="neg_mean_squared_error")
        plscv_err.append(-1*np.mean(plsrcv["test_score"]))
        print(i, np.mean(plsrcv["test_score"]))

        pca_cv = PCA(n_components=i)
        pca_cv.fit(xdata)
        pca_scores = pca_cv.transform(xdata)

        pcr_cv = LinearRegression().fit(pca_scores, ydata)
        pcacv  = cross_validate(pcr_cv, pca_scores, ydata, cv=10, scoring="neg_mean_squared_error")
        pcrcv_err.append(-1*np.mean(pcacv["test_score"]))
        print(i, np.mean(pcacv["test_score"]))


    # Plotting section

    f, axs = plt.subplots(3,3)

#    ax.plot([np.sum(fractions_of_explained_variance[0:i]) for i in range(len(fractions_of_explained_variance))],'-bo')
    axs[0,0].plot(range(1,11),fev,'-bo')
    #axs[0,0].plot(range(1,11),fevc,'-gx')
    axs[0,0].set_xlabel("Number of PLS Components")
    axs[0,0].set_ylabel("Percent Variance Explained in Y")

    axs[0,1].plot(ydata, y2fit, ' ob')
    axs[0,1].plot(ydata, yP2fit, ' ^r')
    axs[0,1].plot(range(83,91), range(83,91), ':', color='#888888')
    axs[0,1].set_xlim(83,90)
    axs[0,1].set_ylim(83,90)
    axs[0,1].set_xlabel("Observed Response")
    axs[0,1].set_ylabel("Fitted Response")
    axs[0,1].legend(["PLSR with 2 Components  (R2 = %2.4f)" % (r2PLS2),
                "PCR with 2 Components  (R2 = %2.4f)" % (r2PCR2)],
                prop={'size': 8})

    axs[1,0].plot(range(1,11), 100*np.cumsum(fractions_of_explained_variance)/np.sum(fractions_of_explained_variance), "-ob")
    axs[1,0].plot(range(1,11), 100*(np.cumsum(pca.explained_variance_ratio_)/np.sum(pca.explained_variance_ratio_)), '-^r')
    axs[1,0].set_xlabel("Number of PLS/PCR Components")
    axs[1,0].set_ylabel("Percent Variance Explained in X")
    axs[1,0].legend(["PLSR", "PCR"],
                    prop={'size': 8})

    axs[0,2].plot(ydata, yfit, ' ob')
    axs[0,2].plot(ydata, yPfit, ' ^r')
    axs[0,2].plot(range(83,91), range(83,91), ':', color='#888888')
    axs[0,2].set_xlim(83,90)
    axs[0,2].set_ylim(83,90)
    axs[0,2].set_xlabel("Observed Response")
    axs[0,2].set_ylabel("Fitted Response")
    axs[0,2].legend(["PLSR (10 Components)  (R2 = %2.4f)" % (r2PLS),
                "PCR with (10 Components)  (R2 = %2.4f)" % (r2PCR)],
                prop={'size': 8})

    axs[1,1].plot(range(1,11), plscv_err, '-ob')
    axs[1,1].plot(range(1,11), pcrcv_err, '-^r')
    axs[1,1].legend(["PLSR", "PCR"],
                    prop={'size': 8})
    axs[1,1].set_ylabel("Estimated Mean Squared Prediction Error")
    axs[1,1].set_xlabel("Number of PLS/PCR Components")

    axs[1,2].plot(pls3.x_weights_)
    axs[1,2].set_xlabel("Variable")
    axs[1,2].set_xlabel("PLS Weight")
    axs[1,2].legend(["1st Component",
                     "2nd Component",
                     "3rd Component"],
                     prop={'size': 8})

    axs[2,2].plot(pca.components_.T[:,0:4])
    axs[2,2].set_xlabel("Variable")
    axs[2,2].set_xlabel("PCA Loading")
    axs[2,2].legend(["1st Component",
                     "2nd Component",
                     "3rd Component",
                     "4th Component"],
                     prop={'size': 8})

    axs[2,0].plot(xdata.values.T)
    axs[2,0].set_title("Dataset " + dataset)

    axs[2,1].plot(xdata_norm.values.T)
    axs[2,1].set_title("Dataset " + dataset + " (mean-normalised)")

    plt.show()
