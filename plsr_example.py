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
    dataset = "octane"
    wls, xdata, ydata = import_dataset(dataset)

    total_variance_in_y = np.var(ydata, axis = 0)
    nc=10

    pls = PLSRegression(n_components=nc,)
    pls2 = PLSRegression(n_components=2)
    pls3 = PLSRegression(n_components=3)

    pca = PCA(n_components=nc)
    pca2 = PCA(n_components=2)

    print("X shape: %s, Y shape: %s" % (np.shape(xdata),np.shape([ydata])))
    print("Fitting PLS...")
    pls.fit(xdata, ydata)
    pls2.fit(xdata, ydata)
    pls3.fit(xdata, ydata)
    print("Fitting PCA...")
    pca.fit(xdata)
    pca2.fit(xdata)

    yfit = pls.predict(xdata)
    y2fit = pls2.predict(xdata)

    pca_scores = pca.transform(xdata)
    pca2_scores = pca2.transform(xdata)

    pcr = LinearRegression().fit(pca_scores, ydata)
    pcr2 = LinearRegression().fit(pca2_scores, ydata)

    yPfit = pcr.predict(pca.transform(xdata))
    yP2fit = pcr2.predict(pca2.transform(xdata))

    variance_in_y = np.var(pls.y_scores_, axis = 0)
    fractions_of_explained_variance = variance_in_y / total_variance_in_y

    TSS = np.sum((ydata-np.mean(ydata))**2 )

    RSS_PLS2 = np.sum(np.subtract(ydata,y2fit[:,0])**2 )
    r2PLS2 = 1 - RSS_PLS2/TSS

    RSS_PCR2 = np.sum((ydata-yP2fit)**2 )
    r2PCR2 = 1 - RSS_PCR2/TSS

    RSS_PLS = np.sum(np.subtract(ydata,yfit[:,0])**2 )
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

    total_variance_in_x = np.sum(np.var(xdata, axis = 0))

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

    # normalize variance by total variance:
    fractions_of_explained_variance = np.sum(variance_in_x / total_variance_in_x, axis=1)

    '''
    CROSS VALIDATION STAGE
    '''
    plscv_err = []
    pcrcv_err = []

    xdata_norm = xdata-np.mean(xdata)

    for i in range(1,nc+1):
        pls_cv = PLSRegression(n_components=i)
        plsrcv  = cross_validate(pls_cv, xdata, ydata, cv=10, scoring="neg_mean_squared_error")
        plscv_err.append(-1*np.mean(plsrcv["test_score"]))

        pca_cv = PCA(n_components=i)
        pca_cv.fit(xdata)
        pca_scores = pca_cv.transform(xdata)

        pcr_cv = LinearRegression().fit(pca_scores, ydata)
        pcacv  = cross_validate(pcr_cv, pca_scores, ydata, cv=10, scoring="neg_mean_squared_error")
        pcrcv_err.append(-1*np.mean(pcacv["test_score"]))

    '''
    Plotting section
    '''
    # Axes for dataset visualisation
    f_dataset, axs_ds = plt.subplots(1,2)
    ds_ax = axs_ds[0]
    dsc_ax = axs_ds[1]
    # Axes for PLSR PCR comparisons
    f_comp, axs_comp = plt.subplots(2,2)
    yvar_ax = axs_comp[0,0]
    xvar_ax = axs_comp[0,1]
    comp2_ax = axs_comp[1,0]
    comp10_ax = axs_comp[1,1]
    # Axes for results visualisation
    f_res, axs_res = plt.subplots(3,1)
    mse_ax = axs_res[0]
    plsc_ax = axs_res[1]
    pcrc_ax = axs_res[2]

    # Plot dataset (raw)
    ds_ax.plot(wls, xdata.values.T, linewidth=0.5)
    ds_ax.set_title("Dataset " + dataset)
    ds_ax.set_xlabel("wavelength [nm]")
    # Plot dataset (mean centered)
    dsc_ax.plot(wls, xdata_norm.values.T, linewidth=0.5)
    dsc_ax.set_title("Dataset " + dataset + " (mean-normalised)")
    dsc_ax.set_xlabel("wavelength [nm]")
    # Plot Variance in Y
    yvar_ax.plot(range(1,11),fev,'-bo',fillstyle='none', linewidth=0.5)
    yvar_ax.set_xlabel("PLS Components")
    yvar_ax.set_ylabel("Variance Explained in Y [%]")
    yvar_ax.set_xlim(1,10)
    yvar_ax.set_ylim(0.30,1.00)
    yvar_ax.legend(["PLSR"],
                    prop={'size': 8})
    # Plot Variance in X
    xvar_ax.plot(range(1,11), 100*np.cumsum(fractions_of_explained_variance)/np.sum(fractions_of_explained_variance), "-ob")
    xvar_ax.plot(range(1,11), 100*(np.cumsum(pca.explained_variance_ratio_)/np.sum(pca.explained_variance_ratio_)), '-^r')
    xvar_ax.set_xlabel("PLS/PCR Components")
    xvar_ax.set_ylabel("Variance Explained in X [%]")
    xvar_ax.legend(["PLSR", "PCR"],
                    prop={'size': 8})
    # Plot comparison with 2 components
    comp2_ax.plot(ydata, y2fit, ' ob')
    comp2_ax.plot(ydata, yP2fit, ' ^r')
    comp2_ax.plot(range(83,91), range(83,91), ':', color='#888888')
    comp2_ax.set_xlim(83,90)
    comp2_ax.set_ylim(83,90)
    comp2_ax.set_xlabel("Observed Response")
    comp2_ax.set_ylabel("Fitted Response")
    comp2_ax.legend(["PLSR (2 Comp.) (R2: %2.3f)" % (r2PLS2),
                "PCR (2 Comp.)(R2: %2.3f)" % (r2PCR2)],
                prop={'size': 8})
    # Plot comparison with 10 components
    comp10_ax.plot(ydata, yfit, ' ob')
    comp10_ax.plot(ydata, yPfit, ' ^r')
    comp10_ax.plot(range(83,91), range(83,91), ':', color='#888888')
    comp10_ax.set_xlim(83,90)
    comp10_ax.set_ylim(83,90)
    comp10_ax.set_xlabel("Observed Response")
    comp10_ax.set_ylabel("Fitted Response")
    comp10_ax.legend(["PLSR (10 Comp.) (R2: %2.3f)" % (r2PLS),
                "PCR (10 Comp.)  (R2: %2.3f)" % (r2PCR)],
                prop={'size': 8})
    # Plot comparison of MSE
    mse_ax.plot(range(1,11), plscv_err, '-ob')
    mse_ax.plot(range(1,11), pcrcv_err, '-^r')
    mse_ax.legend(["PLSR", "PCR"],
                    prop={'size': 8})
    mse_ax.set_ylabel("Estimated Mean\nSquared Prediction Error")
    mse_ax.set_xlabel("Number of PLS/PCR Components")
    # Plot PCA component weights
    plsc_ax.plot(pls3.x_weights_)
    plsc_ax.set_xlabel("Variable")
    plsc_ax.set_ylabel("PLS Weight")
    plsc_ax.legend(["1st Component",
                     "2nd Component",
                     "3rd Component"],
                     prop={'size': 8})
    # Plot PLSR component loadings
    pcrc_ax.plot(pca.components_.T[:,0:4])
    pcrc_ax.set_xlabel("Variable")
    pcrc_ax.set_ylabel("PCA Loading")
    pcrc_ax.legend(["1st Component",
                     "2nd Component",
                     "3rd Component",
                     "4th Component"],
                     prop={'size': 8})
    plt.show()
