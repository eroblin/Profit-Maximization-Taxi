# Profit-Maximization-Taxi
We try to maximise the profit of a taxi driver. We try to understand which factors explain this profit level. We define two situations:
- when the taxi has informations before the ride
- when the taxi has no informations.

We use a regression to explain the profit. The data is composed by 170 millions of rides for the year 2013 in New York City. It is extracted from the website: http://www.andresmh.com/nyctaxitrips/. There is two tables per month ("trip" and "fare), each one of 2 Go. We use a virtual machine with AZURE to deal with the datasets. We downoad them on an SQL server and use a python cursor. The cursor is a way to evaluate an SQL request without storing any data in memory.
To realise our different regressions, we define:
- the dependant variable: we use two different ways of defining the profit. First, the profit is the total amount of the ride. Secondly, the profit is seen as the tip. Therefore, we have two different regressions with two different explanatory variables ;
- the explanatory variables: start time of the ride, number of passengers, trip distance, trip time, departure longitude, departure latitude.

We join the two tables. Then, we sample randmply 100 000 rides per month to build a representative database. We sample again randomly 50 000 lines inside this dataset. Finaly, we only select the data for one day (the "monday"), to avoid biases in the results due to the differences in the days. We still keep a time variable, the hour, to compare the profit realised during the day. We split the dataset in two sets: the training and the test set.
We build a variable to analyse the location of the different rides. They are mainly located in Manhattan.


