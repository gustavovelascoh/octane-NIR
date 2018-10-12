import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score


FILENAME = "octane.xlsx"


if __name__ == "__main__":
    
    data = pandas.read_excel(FILENAME)
    
    print(type(data), np.shape(data))
    print(data.loc[0:3,'Octane number':'1108']) 

    wls = np.array([ int(i) for i in data.columns.values[2:]])
    print(wls)
    wls_len = len(wls)

    x_data = data.loc[:,'1100':]
    y_data = data['Octane number']

    total_variance_in_y = np.var(y_data, axis = 0)
    nc=10


    pls2 = PLSRegression(n_components=nc)
    pls2.fit(x_data, y_data)

    variance_in_y = np.var(pls2.y_scores_, axis = 0)

    print("xw ", pls2.x_weights_)
    print("yw ", pls2.y_weights_)
    print("xl ", pls2.x_loadings_)
    print("yl ", pls2.y_loadings_)
    print("yl ", pls2.y_scores_)

    fractions_of_explained_variance = variance_in_y / total_variance_in_y

    r2_sum = 0
    
    fev = []

    for i in range(0,nc):
        Y_pred=np.dot(pls2.x_scores_[:,i].reshape(-1,1),
                        pls2.y_loadings_[:,i].reshape(-1,1).T) * y_data.std(axis=0, ddof=1) + y_data.mean(axis=0)
        r2_sum += round(r2_score(y_data,Y_pred),3) 
        fev.append(r2_sum)
        print('R2 for %d component: %g' %(i+1,round(r2_score(y_data,Y_pred),3)))

    print('R2 for all components (): %g' %r2_sum) #Sum of above

    ax = plt.axes()

#    ax.plot([np.sum(fractions_of_explained_variance[0:i]) for i in range(len(fractions_of_explained_variance))],'-bo')
    ax.plot(fev,'-bo')
    ax.set_xlabel("Number of PLS Components")
    ax.set_ylabel("Percent Variance Explained in Y")

    plt.show()

    exit()
    ax = plt.axes(projection='3d')

    for idx, rdata in data.iterrows():
        on = rdata['Octane number']
        vals = rdata.loc['1100':]
        print(on)
        on_ar = np.full(wls_len, on)
        
        print(on_ar, wls, vals)
        
        ax.plot(on_ar, wls, vals)

    ax.set_xlabel("Octane number")
    ax.set_ylabel("Wavelength")
    ax.set_zlabel("Value")
    plt.show()
    
    exit()


    ax.plot(data.loc[:,'Octane number'], data.loc[:,'1100':].T, ' .')
    plt.show()
