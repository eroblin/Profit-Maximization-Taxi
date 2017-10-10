################# Data preparation #############################################################
%pylab inline
from fonctions import *


ntot = 100000
nrand=50000
#Table with mondays from 1 am to 11 pm included. Y : profit as total amount / Z : tip/ X : explanatory variables
X_tot,Y_tot,Z_tot = variables(1, 'Monday',1,23, ntot, nrand)

for i in range(2,13):
    X,Y,Z = variables(i, 'Monday',1,23, ntot, nrand)
    X_tot =np.concatenate((X,X_tot))
    Y_tot = np.concatenate((Y,Y_tot))
    Z_tot = np.concatenate((Z,Z_tot))
    print("We selected ",len(Y_tot), " rides yet.")
                            
print(X_tot.shape)

#1 = Manhattan
#2 = Bronx
#3 = Queens
#4 = Brooklyn
#5 = Staten Island
#0 = None of the above

for i in range(len(X_tot)) :
    long=X_tot[i][4]
    lati=X_tot[i][5]
    if long>=-73.936854 and long<=-73.814991 and lati >=40.813647 and lati <=40.894584 :
        X_tot[i][6]=2
        print('Bronx')
    elif long>=-73.936854 and long<=-73.751520 and lati>=40.704819 and lati <= 40.775766 :
        X_tot[i][6]=3
        print('Queens')
    elif long >=-74.041629 and long <= -73.858295 and lati >=40.572734 and lati <=40.704819 :
        X_tot[i][6]=4
        print('Brooklyn')
    elif long >= -74.222290 and long<= -74.054728 and lati >= 40.510196 and lati <= 40.68754 :
        X_tot[i][6]=5
        print('Staten Island')
    elif long>= -74.017371 and long<=-73.936854 and lati>=40.704819 and lati<=40.813647 :
        X_tot[i][6]=1
        #print('Manhattan')
    else :
        X_tot[i][6]=0
        
#train and test set    
trX_tot=np.transpose(X_tot)  
X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X_tot, Y_tot, Z_tot, test_size=0.33)

################################### Situation 1 #########################################################

#1. Regression: profit explained by trip time, trip distance and start time of the trip
X_train_reg1= np.transpose([np.transpose(X_train)[i] for i in (0,2,3)]) 
X_test_reg1 = np.transpose([np.transpose(X_test)[i] for i in (0,2,3)]) 

errteY=[] #Error on test set
errtrY=[] #Error on train set
errteZ=[] 
errtrZ=[]
#We try different lengths of leaf
for i in range(1,20): 
    clf1Y=DecisionTreeRegressor(max_depth=i).fit(X_train_reg1, Y_train)
    Y_test_pred, Y_train_pred=clf1Y.predict(X_test_reg1), clf1Y.predict(X_train_reg1)
    errtrY+=[mean_squared_error(Y_train_pred, Y_train)**0.5]
    errteY+=[mean_squared_error(Y_test_pred, Y_test)**0.5]
    clf1Z=DecisionTreeRegressor(max_depth=i).fit(X_train_reg1, Z_train)
    Z_test_pred, Z_train_pred=clf1Z.predict(X_test_reg1), clf1Z.predict(X_train_reg1)
    errtrZ+=[mean_squared_error(Z_train_pred, Z_train)**0.5]
    errteZ+=[mean_squared_error(Z_test_pred, Z_test)**0.5]

plt.figure(0, figsize=(15,5))
plt.subplot(1,2,1)
plt.plot([i for i in range(1,20)], errtrY, label='trainY')
plt.plot([i for i in range(1,20)], errteY, label='testY')
plt.ylabel('Erreur pour le montant total')
plt.xlabel("Longueur de l'arbre")
plt.ylim(0,7.5)
plt.xlim(0,20)
plt.legend()
plt.subplot(1,2,2)
plt.plot([i for i in range(1,20)], errtrZ, label='trainZ')
plt.plot([i for i in range(1,20)], errteZ, label='testZ')
plt.ylabel('Error for the amount of the tip')
plt.xlabel("Length of the tree")
plt.ylim(0,3)
plt.xlim(0,20)
plt.legend()

max_depthY=7
max_depthZ=5
clf1Y=DecisionTreeRegressor(max_depth=max_depthY).fit(X_train_reg1, Y_train)
clf1Z=DecisionTreeRegressor(max_depth=max_depthZ).fit(X_train_reg1, Z_train)
Y_train_pred, Y_test_pred = clf1Y.predict(X_train_reg1), clf1Y.predict(X_test_reg1)
Z_train_pred, Z_test_pred = clf1Z.predict(X_train_reg1), clf1Z.predict(X_test_reg1)

print("Variables importance",clf1Y.feature_importances_,"\n R square",  clf1Y.score(X_train_reg1,Y_train))
print("Variables importance",clf1Z.feature_importances_,"\n R square",  clf1Z.score(X_train_reg1,Z_train))


# 2. Regression: trip distance explained by longitude, latitude and number of passengers
X_train_reg2= np.transpose([np.transpose(X_train)[i] for i in (1,4,5)]) #passenger_count, longitude, latitude
X_test_reg2 = np.transpose([np.transpose(X_test)[i] for i in (1,4,5)])
T_train=np.transpose(np.transpose(X_train)[2])   #trip_time
T_test=np.transpose(np.transpose(X_test)[2])
D_train=np.transpose(np.transpose(X_train)[3])   #trip_distance
D_test=np.transpose(np.transpose(X_test)[3])

errteT=[]
errtrT=[]
errteD=[]
errtrD=[]

for i in range(1,20): 
    clf2T=DecisionTreeRegressor(max_depth=i).fit(X_train_reg2, T_train)
    clf2D=DecisionTreeRegressor(max_depth=i).fit(X_train_reg2, D_train)
    T_train_pred, T_test_pred=clf2T.predict(X_train_reg2), clf2T.predict(X_test_reg2)
    D_train_pred, D_test_pred=clf2D.predict(X_train_reg2), clf2D.predict(X_test_reg2)
    errtrT+=[mean_squared_error(T_train_pred, T_train)**0.5]
    errteT+=[mean_squared_error(T_test_pred, T_test)**0.5]
    errtrD+=[mean_squared_error(D_train_pred, D_train)**0.5]
    errteD+=[mean_squared_error(D_test_pred, D_test)**0.5]

plt.figure(0, figsize=(15,5))
plt.subplot(1,2,1)
plt.plot([i for i in range(1,20)], errtrT, label='trainT')
plt.plot([i for i in range(1,20)], errteT, label='testT')
plt.ylabel('Error for the trip time')
plt.xlabel("Length of the tree")
#plt.ylim(480,520)
plt.xlim(0,20)
plt.legend()
plt.subplot(1,2,2)
plt.plot([i for i in range(1,20)], errtrD, label='trainD')
plt.plot([i for i in range(1,20)], errteD, label='testD')
plt.ylabel('Error for the trip time')
plt.xlabel("Length of the tree")
#plt.ylim(480,520)
plt.xlim(0,20)
plt.legend()


max_depthT = 10 
max_depthD = 7
clf2T=DecisionTreeRegressor(max_depth=max_depthT).fit(X_train_reg2, T_train)
clf2D=DecisionTreeRegressor(max_depth=max_depthD).fit(X_train_reg2, D_train)
T_train_pred, T_test_pred=clf2T.predict(X_train_reg2), clf2T.predict(X_test_reg2)
D_train_pred, D_test_pred=clf2D.predict(X_train_reg2), clf2D.predict(X_test_reg2)

print("Variables importance",clf2T.feature_importances_,"\n R square",  clf2T.score(X_train_reg2,T_train))
print("Variables importance",clf2D.feature_importances_,"\n R square",  clf2D.score(X_train_reg2,D_train))

################ Situation 2 ########################################################

########## Decision Tree ###############

# 1. Régression 2.1 : amount explained by trip time, trip distance, start time, passengers count
X_train_reg6= np.transpose([np.transpose(X_train)[i] for i in (0,1,2,3)])
X_test_reg6 = np.transpose([np.transpose(X_test)[i] for i in (0,1,2,3)]) 
errteY=[] # Error on test set
errtrY=[] # Error on training set
errteZ=[] 
errtrZ=[]

#parameter optimization
for i in range(1,20): 
    clf6Y=DecisionTreeRegressor(max_depth=i).fit(X_train_reg6, Y_train)
    Y_test_pred, Y_train_pred=clf6Y.predict(X_test_reg6), clf6Y.predict(X_train_reg6)
    errtrY+=[mean_squared_error(Y_train_pred, Y_train)**0.5]
    errteY+=[mean_squared_error(Y_test_pred, Y_test)**0.5]
    clf6Z=DecisionTreeRegressor(max_depth=i).fit(X_train_reg6, Z_train)
    Z_test_pred, Z_train_pred=clf6Z.predict(X_test_reg6), clf6Z.predict(X_train_reg6)
    errtrZ+=[mean_squared_error(Z_train_pred, Z_train)**0.5]
    errteZ+=[mean_squared_error(Z_test_pred, Z_test)**0.5]

plt.figure(0, figsize=(15,5))
plt.subplot(1,2,1)
plt.plot([i for i in range(1,20)], errtrY, label='trainY')
plt.plot([i for i in range(1,20)], errteY, label='testY')
plt.ylabel('Error for the total amount')
plt.xlabel("Length of the tree")
plt.ylim(0,7.5)
plt.xlim(0,20)
plt.legend()
plt.subplot(1,2,2)
plt.plot([i for i in range(1,20)], errtrZ, label='trainZ')
plt.plot([i for i in range(1,20)], errteZ, label='testZ')
plt.ylabel('Error for the tip')
plt.xlabel("Length of the tree")
plt.ylim(0,3)
plt.xlim(0,20)
plt.legend()

max_depthY = 10
max_depthZ = 5
clf6Y=DecisionTreeRegressor(max_depth=max_depthY).fit(X_train_reg6, Y_train)
clf6Z=DecisionTreeRegressor(max_depth=max_depthZ).fit(X_train_reg6, Z_train)
Y_train_pred, Y_test_pred = clf6Y.predict(X_train_reg6), clf6Y.predict(X_test_reg6)
Z_train_pred, Z_test_pred = clf6Z.predict(X_train_reg6), clf6Z.predict(X_test_reg6)

print("Variables importance",clf6Y.feature_importances_,"\n R square",  clf6Y.score(X_train_reg6,Y_train))
print("Variables importance",clf6Z.feature_importances_,"\n R square",  clf6Z.score(X_train_reg6,Z_train))

#2. Régression:distance explained by longitude, latitude 
X_train_reg7= np.transpose([np.transpose(X_train)[i] for i in (4,5)]) #longitude, latitude
X_test_reg7 = np.transpose([np.transpose(X_test)[i] for i in (4,5)])
T_train=np.transpose(np.transpose(X_train)[2])   #trip_time
T_test=np.transpose(np.transpose(X_test)[2])
D_train=np.transpose(np.transpose(X_train)[3])   #trip_distance
D_test=np.transpose(np.transpose(X_test)[3])
errteT=[]
errtrT=[]
errteD=[]
errtrD=[]

for i in range(1,20): 
    clf7T=DecisionTreeRegressor(max_depth=i).fit(X_train_reg7, T_train)
    clf7D=DecisionTreeRegressor(max_depth=i).fit(X_train_reg7, D_train)
    T_train_pred, T_test_pred=clf7T.predict(X_train_reg7), clf7T.predict(X_test_reg7)
    D_train_pred, D_test_pred=clf7D.predict(X_train_reg7), clf7D.predict(X_test_reg7)
    errtrT+=[mean_squared_error(T_train_pred, T_train)**0.5]
    errteT+=[mean_squared_error(T_test_pred, T_test)**0.5]
    errtrD+=[mean_squared_error(D_train_pred, D_train)**0.5]
    errteD+=[mean_squared_error(D_test_pred, D_test)**0.5]


plt.figure(0, figsize=(15,5))
plt.subplot(1,2,1)
plt.plot([i for i in range(1,20)], errtrT, label='trainT')
plt.plot([i for i in range(1,20)], errteT, label='testT')
plt.ylabel("Erreur for the trip time")
plt.xlabel("Length of the tree")
#plt.ylim(480,520)
plt.xlim(0,20)
plt.legend()
plt.subplot(1,2,2)
plt.plot([i for i in range(1,20)], errtrD, label='trainD')
plt.plot([i for i in range(1,20)], errteD, label='testD')
plt.ylabel("Erreur for the trip distance")
plt.ylabel("Erreur for the trip time")
plt.xlabel("Length of the tree")
#plt.ylim(480,520)
plt.xlim(0,20)
plt.legend()


max_depthT = 10 
max_depthD = 8
clf7T=DecisionTreeRegressor(max_depth=max_depthT).fit(X_train_reg7, T_train)
clf7D=DecisionTreeRegressor(max_depth=max_depthD).fit(X_train_reg7, D_train)
T_train_pred, T_test_pred=clf7T.predict(X_train_reg7), clf7T.predict(X_test_reg7)
D_train_pred, D_test_pred=clf7D.predict(X_train_reg7), clf7D.predict(X_test_reg7)

print("Variables importance",clf7T.feature_importances_,"\n R square",  clf7T.score(X_train_reg7,T_train))
print("Variables importance",clf7D.feature_importances_,"\n R square",  clf7D.score(X_train_reg7,D_train))

###### Random Forest #####################################################
# 1. Régression 2.1 : amount explained by trip time, trip distance, start time, passengers count
X_train_reg3= np.transpose([np.transpose(X_train)[i] for i in (0,1,2,3)])
#heure, passenger_count, triptime, tripdistance
X_test_reg3 = np.transpose([np.transpose(X_test)[i] for i in (0,1,2,3)]) 
errteY=[] #Error on test set
errtrY=[] #Error on train set
errteZ=[] 
errtrZ=[]

for n_estimators in range(1,30): 
    clf3Y=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depthY,min_samples_leaf=10).fit(X_train_reg3, Y_train)
    Y_test_pred, Y_train_pred=clf3Y.predict(X_test_reg3), clf3Y.predict(X_train_reg3)
    errtrY+=[mean_squared_error(Y_train_pred, Y_train)**0.5]
    errteY+=[mean_squared_error(Y_test_pred, Y_test)**0.5]
    clf3Z=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depthZ,min_samples_leaf=10).fit(X_train_reg3, Z_train)
    Z_test_pred, Z_train_pred=clf3Z.predict(X_test_reg3), clf3Z.predict(X_train_reg3)
    errtrZ+=[mean_squared_error(Z_train_pred, Z_train)**0.5]
    errteZ+=[mean_squared_error(Z_test_pred, Z_test)**0.5]


plt.figure(0, figsize=(15,5))
plt.subplot(1,2,1)
plt.plot([i for i in range(1,30)], errtrY, label='trainY')
plt.plot([i for i in range(1,30)], errteY, label='testY')
plt.ylabel('Error for the total amount')
plt.xlabel("Number of trees in the forest")
plt.ylim(0,7.5)
plt.xlim(0,20)
plt.legend()
plt.subplot(1,2,2)
plt.plot([i for i in range(1,30)], errtrZ, label='trainZ')
plt.plot([i for i in range(1,30)], errteZ, label='testZ')
plt.ylabel('Error for the tip')
plt.xlabel("Number if trees in the forest")
plt.ylim(0,3)
plt.xlim(0,20)
plt.legend()

n_estimatorsY=7
n_estimatorsZ=5
clf3Y=RandomForestRegressor(n_estimators=n_estimatorsY,max_depth=max_depthY,min_samples_leaf=10).fit(X_train_reg3, Y_train)
clf3Z=RandomForestRegressor(n_estimators=n_estimatorsZ,max_depth=max_depthZ,min_samples_leaf=10).fit(X_train_reg3, Z_train)
Y_train_pred, Y_test_pred = clf3Y.predict(X_train_reg3), clf3Y.predict(X_test_reg3)
Z_train_pred, Z_test_pred = clf3Z.predict(X_train_reg3), clf3Z.predict(X_test_reg3)

print("Variables importance",clf3Y.feature_importances_,"\n R square",  clf3Y.score(X_train_reg3,Y_train))
print("Variables importance",clf3Z.feature_importances_,"\n R square",  clf3Z.score(X_train_reg3,Z_train))

# 2. Regression: distance explained longitude, latitude
X_train_reg4= np.transpose([np.transpose(X_train)[i] for i in (4,5)]) #longitude, latitude
X_test_reg4 = np.transpose([np.transpose(X_test)[i] for i in (4,5)])
T_train=np.transpose(np.transpose(X_train)[2])   #trip_time
T_test=np.transpose(np.transpose(X_test)[2])
D_train=np.transpose(np.transpose(X_train)[3])   #trip_distance
D_test=np.transpose(np.transpose(X_test)[3])
errteT=[]
errtrT=[]
errteD=[]
errtrD=[]

for n_estimators in range(1,20): 
    clf4T=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depthT,min_samples_leaf=10).fit(X_train_reg4, T_train)
    clf4D=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depthD,min_samples_leaf=10).fit(X_train_reg4, D_train)
    T_train_pred, T_test_pred=clf4T.predict(X_train_reg4), clf4T.predict(X_test_reg4)
    D_train_pred, D_test_pred=clf4D.predict(X_train_reg4), clf4D.predict(X_test_reg4)
    errtrT+=[mean_squared_error(T_train_pred, T_train)**0.5]
    errteT+=[mean_squared_error(T_test_pred, T_test)**0.5]
    errtrD+=[mean_squared_error(D_train_pred, D_train)**0.5]
    errteD+=[mean_squared_error(D_test_pred, D_test)**0.5]


plt.figure(0, figsize=(15,5))
plt.subplot(1,2,1)
plt.plot([i for i in range(1,20)], errtrT, label='trainT')
plt.plot([i for i in range(1,20)], errteT, label='testT')
plt.ylabel("Error for the trip time (in sec)")
plt.xlabel("Number of trees in the forest")
#plt.ylim(480,520)
plt.xlim(0,20)
plt.legend()
plt.subplot(1,2,2)
plt.plot([i for i in range(1,20)], errtrD, label='trainD')
plt.plot([i for i in range(1,20)], errteD, label='testD')
plt.ylabel("Error for the trip distance")
plt.xlabel("Number of trees in the forest")
#plt.ylim(480,520)
plt.xlim(0,20)
plt.legend()

n_estimatorsY=15
n_estimatorsZ=15
clf4Y=RandomForestRegressor(n_estimators=n_estimatorsY,max_depth=max_depthY,min_samples_leaf=10).fit(X_train_reg4, Y_train)
clf4Z=RandomForestRegressor(n_estimators=n_estimatorsZ,max_depth=max_depthZ,min_samples_leaf=10).fit(X_train_reg4, Z_train)
Y_train_pred, Y_test_pred = clf4Y.predict(X_train_reg4), clf4Y.predict(X_test_reg4)
Z_train_pred, Z_test_pred = clf4Z.predict(X_train_reg4), clf4Z.predict(X_test_reg4)

print("Variables importance",clf4Y.feature_importances_,"\n R square",  clf4Y.score(X_train_reg4,Y_train))
print("Variables importance",clf4Z.feature_importances_,"\n R square",  clf4Z.score(X_train_reg4,Z_train))

############### MLP ##################################################################
X_train_reg5= np.transpose([np.transpose(X_train)[i] for i in (0,1,2,3)]) #hour, passenger_count, triptime, tripdistance
X_test_reg5 = np.transpose([np.transpose(X_test)[i] for i in (0,1,2,3)])

robust_scaler = sklearn.preprocessing.RobustScaler()
X_train_scaled = robust_scaler.fit_transform(X_train_reg5)
X_test_scaled = robust_scaler.transform(X_test_reg5)

errtrY=[]
errteY=[]
errtrZ=[]
errteZ=[]

for i in range(1,5) :
    clf5Y = Regressor(layers=[Layer("Sigmoid", units = 10*i ),Layer('Linear')],learning_rate=0.02,n_iter=10).fit(X_train_scaled, Y_train)
    clf5Z = Regressor(layers=[Layer("Sigmoid", units = 10*i ),Layer('Linear')],learning_rate=0.02,n_iter=10).fit(X_train_scaled, Z_train)
    errtrY += [mean_squared_error( clf5Y.predict(X_train_scaled), Y_train)**0.5]
    errteY += [mean_squared_error( clf5Y.predict(X_test_scaled), Y_test)**0.5]
    errtrZ += [mean_squared_error( clf5Z.predict(X_train_scaled), Z_train)**0.5]
    errteZ += [mean_squared_error( clf5Z.predict(X_test_scaled), Z_test)**0.5]

plt.figure(0, figsize=(15,5))
plt.subplot(1,2,1)
plt.plot ( [10*i for i in range(1,5)], errtrY, label="trainY")
plt.plot ( [10*i for i in range(1,5)], errteY, label="testY")
plt.ylabel('Error for the total amount')
plt.xlabel("Number of neurons")
plt.legend()
plt.subplot(1,2,2)
plt.plot ( [10*i for i in range(1,6)], errtrZ, label="trainZ")
plt.plot ( [10*i for i in range(1,6)], errteZ, label="testZ")
plt.ylabel('Error for the tip')
plt.xlabel("Number of neurons")
plt.legend()

unitsY= 50
unitsZ=30

clf5Y = Regressor(layers=[Layer("Sigmoid", units = unitsY),Layer('Linear')],learning_rate=0.02,n_iter=10).fit(X_train_scaled, Y_train)
clf5Z = Regressor(layers=[Layer("Sigmoid", units = unitsZ ),Layer('Linear')],learning_rate=0.02,n_iter=10).fit(X_train_scaled, Z_train)

Y_train_pred, Y_test_pred = clf5Y.predict(X_train_scaled), clf5Y.predict(X_test_scaled)
Z_train_pred, Z_test_pred = clf5Z.predict(X_train_scaled), clf5Z.predict(X_test_scaled)
scoreY = 1 - mean_squared_error(Y_train_pred, Y_train)/ mean_squared_error(Y_train, np.mean(Y_train)*np.ones_like(Y_train))
scoreZ = 1 - mean_squared_error(Z_train_pred, Z_train)/ mean_squared_error(Z_train, np.mean(Z_train)*np.ones_like(Z_train))
print("weight, bias and number of layers",clf5Y.get_params(),"\n R carré",  scoreY)
print("weight, bias and name of each layer",clf5Z.get_params(),"\n R carré",  scoreZ)

#Total amount
plt.figure(0, figsize=(15,5))
plt.subplot(1,2,1)
plt.plot ( Y_test_pred, Y_test)
#Bissect line to compare predicted values and observations 
plt.plot(x,x)
plt.ylabel('Observations')
plt.xlabel("Predicted values")
plt.xlim(0,100)
plt.ylim(0,150)
plt.legend()

#Tip
plt.figure(0, figsize=(15,5))
plt.subplot(1,2,2)
plt.plot ( Z_test_pred, Z_test)
#Bissect line to compare predicted values and observations
plt.plot(x,x)
plt.ylabel('Observations')
plt.xlabel("Predicted values")
plt.xlim(0,100)
plt.ylim(0,150)
plt.legend()

########### Representation on a map #######################################################
plotation([DecisionTreeRegressor(max_depth=10),RandomForestRegressor(n_estimators=10,max_depth=10,min_samples_leaf=10),
           Regressor(layers=[Layer("Sigmoid", units = 50 ),Layer('Linear')],learning_rate=0.02,n_iter=10)])
print("fin plot")
from PIL import Image
Image.open("carte.jpg")
