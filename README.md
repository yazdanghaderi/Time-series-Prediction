# Time-series-Prediction

Today, electrical energy has become one of the basic human needs. For everyday use, such as the use of home-grown equipment, industrial equipment, 
lighting, heating and ventilation, and much more, electrical energy is needed. Forecasting Electricity Consumption is important because of that energy supplier and
energy consumer systems should be managed in efficient way. Electricity supply companies need short-term electricity consumption to carry out daily operations, 
fulfillment of commitments and planning of the transfer of electricity from a power company to commercial, office and residential units. Therefore, 
a precise prediction of short-term forecast electricity consumption is vital for an electricity supply company. By expanding the machine learning algorithms 
and proving their effectiveness, the tendency to use these algorithms in different domains. In the short-term prediction of electrical energy consumption, 
the inputs are the amount of sensors recorded at different places in a home. Therefore, any sensor may be interrupted and This interrupt will cause no amount to be
recorded at that moment. This creates an incomplete set of data. In most of the works on short-term energy consumption prediction, the use of 
an incomplete data set for energy prediction has not been raised. In this dissertation, we provide several models that can predict the amount of energy consumed 
in the next 10 minutes, even if there are a set of data that does not contain a number of data. these models are, in fact, a hybrid model 
based on multi-task Gaussian processes and various predictors such as multi-layer perceptron networks, recurrent neural networks and 
convolutional neural networks with the aim of short-term prediction of electric energy consumption. The approach is, in fact, a supervised approach, 
and its learning is done using the labeled data for a residential home. Finally, MGP-lstm model, which is one of the three proposed models, 
shows better results than other models. For example, by assuming %50 missed data, 
MGP-lstm model could improve the accuracy of the results compared to the lstm model in the RMSE and MAE criteria, by %14.3 and %27.5, respectively.
