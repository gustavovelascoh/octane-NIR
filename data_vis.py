import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas
import matplotlib.pyplot as plt
import re


def plot_ds(ax, wls, xdata, ydata):
    '''
    Plot dataset
    ax: matplotlib Axes3DSubplot
    wls: Numpy ndarray: List of wavelength
    xdata: Pandas DataFrame: Measurements
    ydata: Pandas Series: Octane numbers
    '''

    wls_len = len(wls)
    for idx, rdata in xdata.iterrows():
        on = ydata[idx]
        vals = rdata
        on_ar = np.full(wls_len, on)

        ax.plot(on_ar, wls, vals)

def main():
    FILENAME = "octane.xlsx"
    oct_df = pandas.read_excel(FILENAME)
    FILENAME = "gasoline.csv"
    gas_df = pandas.read_csv(FILENAME)

    # Print a preview of the datasets
    print(type(oct_df), np.shape(oct_df))
    print(oct_df.loc[0:3,:])
    print(type(gas_df), np.shape(gas_df))
    print(gas_df.loc[0:3,:])

    # Create the xdata y ydata (list of wavelengths and octane numbers)
    oct_wls = np.array([ int(i) for i in oct_df.columns.values[2:]])
    oct_xdata = oct_df.loc[:,'1100':]
    oct_ydata = oct_df['Octane number']

    reg = re.compile('([0-9]+)')
    gas_wls = np.array([ int(reg.findall(i)[0]) for i in gas_df.columns.values[1:]])
    gas_xdata = gas_df.loc[:,'NIR.900 nm':]
    gas_ydata = gas_df['octane']

    # Create a figure with two axes for 3D plotting
    f,(ax,ax1) = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))

    # Plot datasets and put some labels
    plot_ds(ax, oct_wls, oct_xdata, oct_ydata)
    ax.set_title("Octane Dataset")
    ax.set_xlabel("Octane number")
    ax.set_ylabel("Wavelength")
    ax.set_zlabel("Value")


    plot_ds(ax1, gas_wls, gas_xdata, gas_ydata)
    ax1.set_title("Gasoline Dataset")
    ax1.set_xlabel("Octane number")
    ax1.set_ylabel("Wavelength")
    ax1.set_zlabel("Value")

    plt.show()

if __name__ == "__main__":
    main()
