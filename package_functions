import scipy
import sklearn
import pandas as pd
import numpy as np
import win32com
import pyensae
import pyodbc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.basemap import Basemap
from PIL import Image

from sknn.mlp import Classifier, Regressor, Layer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

server = 'localhost'
db = 'master'

x_min, x_max = -74.25 , -73.70
y_min, y_max = 40.49 , 40.91

#SQL request to keep the time period and the variables we want
def variables(month,day, hour_min, hour_max, ntot, nrand) :  
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + db + ';Trusted_Connection=yes')
    cursor = conn.cursor() 
    if month==1 or month ==2 :
        a = cursor.execute("""
SELECT  a.[medallion], a.[hack_license], DATENAME(dw, a.[pickup_datetime]) as date,DATEPART(hh,a.[pickup_datetime]) as hour, a.[passenger_count], a.[trip_time_in_secs], a.[trip_distance], a.[pickup_longitude], a.[pickup_latitude], b.[ total_amount], b.[ tip_amount]
FROM [master].[dbo].[trip_data_"""+ str(month) +"""] AS a 
JOIN [master].[dbo].[trip_fare_"""+ str(month) +"""] AS b 
ON  a.[medallion] = b.[medallion] AND a.[hack_license]=b.[ hack_license] AND a.[pickup_datetime]=b.[ pickup_datetime] 
""")
    else :
        a = cursor.execute("""
SELECT  a.[medallion], a.[ hack_license], DATENAME(dw, a.[ pickup_datetime]) as date,DATEPART(hh,a.[ pickup_datetime]) as hour, a.[ passenger_count], a.[ trip_time_in_secs],
a.[ trip_distance], a.[ pickup_longitude], a.[ pickup_latitude], b.[ total_amount], b.[ tip_amount]
FROM [master].[dbo].[trip_data_"""+ str(month) +"""] AS a 
JOIN [master].[dbo].[trip_fare_"""+str(month)+"""] AS b 
ON  a.[medallion] = b.[medallion] AND a.[ hack_license]=b.[ hack_license] AND a.[ pickup_datetime]=b.[ pickup_datetime] 
    """)
    #X for explanatory variables ; Y, Z two different dependent variables. 
    X = []
    Y = []
    Z = []
    
    #Add a line at the cursor
    row=a.fetchone()
    c=0
    
    #Keep only the NYC rides
    while len(X)< ntot:
        row =a.fetchone()
        if row :
            row=[row[0], row[1], row[2], float(row[3]), float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[10])]
            if row[3] <= hour_max and row[3] >= hour_min and row[2]== day and row[5]>0 and row[7]<x_max and row[7]>x_min and row[8]<y_max and row[8]>y_min:
                X += [[row[i] for i in (3,4,5,6,7,8,9)]]
                Y += [row[9]]
                Z +=[row[10]]
        else :
            c+=1
            if c>10 :
                print("end of month")
                break
    ntot=len(X)
    print("With ",ntot, "rides for the month ",month)
    conn.close()
    I = np.random.choice(ntot, nrand) #random list of indexes
    X = [X[i] for i in I] #sample [ hour, passenger_count, triptime, tripdistance, longitude, latitude]
    X= np.array(X)
    Y = [Y[i] for i in I] #target for the total amount
    Y= np.array(Y)
    Z = [Z[i] for i in I] #target for the tip
    Z= np.array(Z)
    nrand=X.shape[0]
    print("We selected randomly ", nrand, " rides")
    
    return X,Y,Z


def into_levels(vect,Nlvl):
    levels=[np.percentile(vect, 100*i/Nlvl) for i in range(1,Nlvl)]
    ylvl=[]
    for y in vect:
        c=0
        for i,lvl in enumerate(levels) :
            if y<lvl :
                ylvl+=[i]
                break
            elif i+1==len(levels):
                ylvl+=[i+1]
    ylvl = np.array(ylvl)
    return ylvl



def plotation(clf_list):
    nlines=len(clf_list)
    plt.figure(figsize=(20, 10*nlines))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    X_train_plot = np.transpose([np.transpose(X_train)[i] for i in (4,5)])
    Nlvl=5
    c=1
    mlp_Reg_type=type(Regressor(layers=[Layer("Rectifier",name="hiddenN" )],learning_rate=0.02,n_iter=10))
    mlp_Cla_type=type(Classifier(layers=[Layer("Rectifier",name="hiddenN" )],learning_rate=0.02,n_iter=10))
    robust_scaler=False
    for _, clf in enumerate(clf_list) :
        if hasattr(clf, "predict_proba") :
            print("Classifieur")
            if type(clf) == mlp_Cla_type :
                robust_scaler = sklearn.preprocessing.RobustScaler()
                X_train_plot_scaled = robust_scaler.fit_transform(X_train_plot)
                clfY=clf.fit(X_train_plot_scaled, into_levels(Y_train, Nlvl))
                clfZ=clf.fit(X_train_plot_scaled, into_levels(Z_train, Nlvl))
            else :
                clfY=clf.fit(X_train_plot, into_levels(Y_train, Nlvl))
                clfZ=clf.fit(X_train_plot, into_levels(Z_train, Nlvl))
        else :
            print("Regresseur")
            if type(clf) == mlp_Reg_type :
                robust_scaler = sklearn.preprocessing.RobustScaler()
                X_train_plot_scaled = robust_scaler.fit_transform(X_train_plot)
                clfY=clf.fit(X_train_plot_scaled, Y_train)
                clfZ=clf.fit(X_train_plot_scaled, Z_train)
            else :
                clfY=clf.fit(X_train_plot, Y_train)
                clfZ=clf.fit(X_train_plot, Z_train)
        for _, clfdata in enumerate([clfY, clfZ]):
            axes=plt.subplot(nlines, 2, c)
            m = Basemap(llcrnrlon=x_min, llcrnrlat=y_min , urcrnrlon=x_max, urcrnrlat=y_max ,resolution='i',projection='cass',lon_0=-74.00597,lat_0=40.71427,ax=axes)
            m.drawcoastlines()
            lons, lats=m.makegrid(100,100)
            x, y=m(lons, lats)
            Z=np.zeros((100,100))
            for l in range(100):
                for p in range(100):
                    LP=np.array([lons[l][p], lats[l][p]])
                    LP=np.array([LP])
                    if robust_scaler != False :
                        LP = robust_scaler.transform(LP)
                    Z[l][p]=clfdata.predict(LP)
            diff=np.max(Z)-np.min(Z)
            cs=m.contourf(x, y, Z,[np.min(Z)+diff*i/Nlvl for i in range(0,Nlvl+1)], cmap=cm, alpha=.8)
            m.colorbar(cs,location='bottom',pad="5%")
            c+=1
        robust_scaler=False
