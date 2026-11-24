# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:45:35 2026

@author: nhand
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ====================== QUADRATIC MODEL ======================
def quadratic_2d(XY, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    x, y = XY
    return c1*x**2*y**2 + c2*x**2*y + c3*x*y**2 + c4*x**2 + c5*y**2 + c6*x*y + c7*x + c8*y + c9

# ====================== MAIN FITTING FUNCTION ======================
def fit_quadratic_from_array(data):
    if data.shape[1] != 9:
        raise ValueError("Data must have exactly 9 columns")
    
    # Extract variables (0-based indexing)
    y = data[:, 1]   # 2nd column
    x = data[:, 7]   # 8th column
    z = data[:, 8]   # 9th column
    
    print(f"Data shape: {data.shape}")
    print("Using x from column 8, y from column 2, z from column 9\n")
    
    # Fit the model
    popt, pcov = curve_fit(quadratic_2d, (x, y), z, 
                          p0=[1, 1, 1, 1, 1, 1, 1, 1, 1], maxfev=5000)
    
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = popt
    
    print("=== Best Fit Quadratic Coefficients ===")
    print(f"c1 (x²y²)     = {c1:.8e}")
    print(f"c2 (x²y)      = {c2:.8e}")
    print(f"c3 (x*y²)     = {c3:.8e}")
    print(f"c4 (x²)       = {c4:.8e}")
    print(f"c5 (y²)       = {c5:.8e}")
    print(f"c6 (x*y)      = {c6:.8e}")
    print(f"c7 (x)        = {c7:.8e}")
    print(f"c8 (y)        = {c8:.8e}")
    print(f"c9 (const)    = {c9:.8e}")
    
    # Evaluate fit quality
    z_pred = quadratic_2d((x, y), *popt)
    rmse = np.sqrt(np.mean((z - z_pred)**2))
    r2 = 1 - np.sum((z - z_pred)**2) / np.sum((z - np.mean(z))**2)
    
    print("\nGoodness of Fit:")
    print(f"RMSE = {rmse:.6e}")
    print(f"R²   = {r2:.6f}")
    
    return {
        'coeffs': {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5, 'c6': c6, 'c7': c7, 'c8': c8, 'c9': c9},
        'func': lambda x_val, y_val: quadratic_2d((x_val, y_val), c1, c2, c3, c4, c5, c6, c7, c8, c9),
        'x': x, 'y': y, 'z': z,
        'z_pred': z_pred,
        'rmse': rmse,
        'r2': r2
    }


# ====================== PLOTTING ======================
def plot_fitted_surface(result):
    """Plot original data points and fitted surface"""
    x, y, z = result['x'], result['y'], result['z']
    func = result['func']
    
    fig = plt.figure(figsize=(12, 6))
    
    # 3D Plot
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data points
    ax.scatter(x, y, z, c='blue', marker='o', s=40, alpha=0.8, label='Original Data')
    
    # Create surface grid
    x_surf = np.linspace(x.min(), x.max(), 50)
    y_surf = np.linspace(y.min(), y.max(), 50)
    X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
    Z_surf = func(X_surf, Y_surf)
    
    # Plot fitted surface
    ax.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.6, cmap='viridis', 
                   edgecolor='none')
    
    ax.set_xlabel('X (Column 2)', fontsize=12)
    ax.set_ylabel('Y (Column 8)', fontsize=12)
    ax.set_zlabel('Z (Column 9)', fontsize=12)
    ax.set_title('Quadratic Surface Fit: z = f(x, y)', fontsize=14)
    
    ax.legend()
    plt.tight_layout()
    plt.show()

def pred_vs_actual(result,testdata):
    from scipy import stats
    x = testdata[:, 7]   # 8th column
    y = testdata[:, 1]   # 2nd column
    Y_test = testdata[:, 8]   # 9th column
    func = result['func']
    pred_test = func(x,y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.ravel(pred_test),np.ravel(Y_test))
    r2_Test = r_value**2
    print("R^2 correlation on the test set = " + str(r2_Test))
    #------------------------------------------------
    # Plot
    import matplotlib.pyplot as plt
    font = {'family' : 'Times New Roman',
            'size'   : 11}
    plt.rc('font', **font)
    fig, axs= plt.subplots(nrows=1, ncols=1, figsize=(3,2.5))
    axs.scatter(pred_test, Y_test,marker='.',s=4)
    axs.plot(np.array([0.5,2.5]),np.array([0.5,2.5]),c='k',lw=1.5)
    axs.grid()
    axs.set_xlim(0.5,2.5)
    axs.set_ylim(0.5,2.5)
    axs.set_xticks([0.5,1.0,1.5,2,2.5])
    axs.set_yticks([0.5,1.0,1.5,2,2.5])
    axs.set_aspect('equal')
    axs.set_xlabel('Predicted $B_M$')
    axs.set_ylabel('Actual $B_M$')
    axs.text(0.55,2.25,'Test set')
    axs.text(1.5,0.75,'R2='+str(np.round(r2_Test,4)))
    fig.tight_layout(pad=1.0)
    plt.show()
# ====================== MAIN ======================
if __name__ == "__main__":
    GMType = "Mix"
    # fd, Td, k1d, GMSetID, TM/T0, y0=DM/D0, TM, betaM, BM
    TrainSetFile = f"../Data/BM_Train_{GMType}_HighDM.txt"
    TestSetFile= f"../Data/BM_Test_{GMType}_HighDM.txt"
    data = np.loadtxt(TrainSetFile)
    TestDat = np.loadtxt(TestSetFile)
    result = fit_quadratic_from_array(data)
    plot_fitted_surface(result)
    pred_vs_actual(result,TestDat)