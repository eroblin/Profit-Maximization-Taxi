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


