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

