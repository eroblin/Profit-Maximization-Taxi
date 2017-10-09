# Profit-Maximization-Taxi

We try to maximise the profit of a taxi driver. We try to understand which factors explain this profit level. We use regression models to explain the profit. 

The data is composed by 170 millions of rides for the year 2013 in New York City. It is extracted from the website: http://www.andresmh.com/nyctaxitrips/. There is two tables per month ("trip" and "fare), each one of 2 Go. We use a virtual machine with AZURE to deal with the datasets. We downoad them on an SQL server and use a python cursor. The cursor is a way to evaluate an SQL request without storing any data in memory.

To realise our different regressions, we define:
- the dependant variable: we use two different ways of defining the profit. First, the profit is the total amount of the ride. Secondly, the profit is seen as the tip. Therefore, we have two different regressions with two different explanatory variables ;
- the explanatory variables: start time of the ride, number of passengers, trip distance, trip time, departure longitude, departure latitude.

We join the two tables. Then, we sample randmply 100 000 rides per month to build a representative database. We sample again randomly 50 000 lines inside this dataset. Finaly, we only select the data for one day (the "monday"), to avoid biases in the results due to the differences in the days. We still keep a time variable, the hour, to compare the profit realised during the day. We split the dataset in two sets: the training and the test set.
We build a variable to analyse the location of the different rides. They are mainly located in Manhattan.

 We define two situations:
- Situation 1: when the taxi has informations before the ride. He knows where to go and the number of passengers. In this case, the profit is explained by 3 variables (start time of the ride, trip time, trip distance). Each of these 3 variables is then explained by 3 variables (number of passengers, latitude and longitude). With these two regression systems one after the other, we want to define the most profitable rides according to the available information. In this situation, the taxi knows at what time of the day he should rather work and which length of rides is the most profitable.
- Situation 2: when the taxi has no informations. The taxi rides in the geographical area he thinks will maximise his profit. Here, the profit is explained by 4 variables (start time of the ride, trip time, trip distance, number of passengers). These 4 variables are explained by the departure latitude and longitude. In this situation, the taxi tries to maximise his profit by riding in the most profitable area. 

Three types of model are used: decision tree, random forest, neural network.

We have to define the different parameters of the models. To do so, we use the training set and the evolution of the error term. Here is an example with the decision tree in the situation 1 and when the profit is define as the total amount of the ride. On the train set, the minimum value of the error is reached when the length of the train is 7: this is the value we choose for this hyperparameter. 

