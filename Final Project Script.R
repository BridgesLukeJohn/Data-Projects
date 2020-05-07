#---------------------- FINAL PROJECT ------------------------------ Luke J. Bridges ------------------------

# Scenario: you are planning to purchase you new home and would like to determine how much should you pay for a house in King 
# County that meets your criteria:
  
# - 4 bedroom
# - 3 bathroom
# - 4,000 sq/ft living
# - 5,000 sq/ft lot
# - Good condition (5+)
# - Good grade (7+)
# - Newer construction (2004+)


## Load libraries
if(!require("randomForest")) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require("corrgram")) install.packages("corrgram", repos = "http://cran.us.r-project.org")
if(!require("caret")) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require("reshape2")) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require("GGally")) install.packages("reshape2", repos = "http://cran.us.r-project.org")

library(corrplot)
library(corrgram)
library(randomForest)
library(GGally)
library(caret)
library(tidyverse)
library(ggplot2)
library(lubridate)
library(hydroGOF)
library(mvtnorm)
library(reshape2)

# change the full path to the file
file = '/Users/lukebridges/OneDrive - UW/Certificate in Data Analytics/Data Mining and Predictive Analytics/Lesson 9 - Final Project/Final Project/kc_house_data.csv'

# Load and Prepare the Data Set
house = read_csv(file)
house_full = read.csv(file)
house = house %>% mutate(waterfront_fac = as.factor(waterfront),
                         view_fac = as.factor(view),
                         zipcode_fac = as.factor(zipcode))
house$id <- NULL
house$zipcode <- NULL
house$date <- NULL


summary(house)
glimpse(house)

# Potential Skewed distributions:
# waterfront, view, sqft_basement, yr_renovated, price, bedrooms, Sqft_living & lot (also the 15 versions of these)

ggpairs(house[,c(3:7,2)])

#display price pust next set of columns from house that reasonably fit on the screen 

ggpairs(select(house_full, bedrooms, bathrooms, sqft_living, floors, price))

# after viewing how skewed price was, we did a log price
house$price = log10(house$price)
house$price = house$price

# dispersment of price:
ggplot(house, aes(x = price)) +
  geom_histogram() 



# sqft living and price is highly correlated as would be expected. also, waterfront and price and sqft basement and price.
# there is also a positive correlation between sqft living and yr built, suggesting houses have been getting bigger over time.

## Checking Relationship between price, bedrooms, bathrooms, sqft_living, sqft lot, and floors
plot1 <- ggpairs(house_full[,c(3:7,2)],
               axisLabels="show")
plot1

## Checking Relationship between price, waterfront, view, condition, grade, and sqft_above
plot2<-ggpairs(house_full[,c(8:12,2)],
               axisLabels = "show")
plot2

## Checking Relationship between price, sqft_basement, yr built, yr_renovated, lat, and long
plot3=ggpairs(house_full[,c(13:17,2)],
              axisLabels="show")
plot3


# ----------------- VARIABLE SELECTION --------------------

# data correlation set:
data_cor<- house[, c(1:18)]

#top correlations
cor_level <- .7
correlationMatrix <- cor(data_cor)
cor_melt <- arrange(melt(correlationMatrix), desc(value))

#hint: you can use %% 2 to remove every other row so we don't get duplicates, given 
#hint: several correlated variables interract with each other given us duplicates from melt
dplyr::filter(cor_melt,  )

#show variables that correlate to price only
cor_level <- .5
price_cor <- dplyr::filter(cor_melt, Var1 == "price" )

# Hi Chris - using melt() is not required for the final project. You're correct that it hasn't been in the coursework, and 
# I'm not sure why it was used in the final project template. No need to spend any more time on it!

#remove variables that logically don't contribute to a good model
house$waterfront <- NULL
house$view <- NULL
house$sqft_living15 <- NULL
house$sqft_lot15 <- NULL

# may remove - may not
house$zipcode_fac <- NULL

glimpse(house)


# The assingmnet instructions said to use the below code, however this is not what we learned in class. Also,
# the instructor did not seperate the data into a training and testing set. So, I will be using a different
# methodology that we learned in class, and will also be creating a training and testing set.
tree_model_2 <- randomForest(price ~., data = house)
summary(tree_model_2)

plot(tree_model_2)

var_imp_treemodel_2 = varImp(tree_model_2)
var_imp_treemodel_2
varImpPlot(tree_model_2)


# I eneded up deciding too try this method as well, it looks like the instructor used this method to decide on which
# variables to use for the later linear regression model, and not to actually predict. I checked this method and
# it the variable importance plots showed a difference in terms of features that should be used for future prediction.
# The in class method showed sqft_living, grade, zipcode_fac, sqft_above, and bathrooms as the most important.
# The final project method showed grade, sqft_living, lat, sqft_above, long, and bathrooms to be the most important.
# Through this exploration, I will be using the final project method to determine my variable choice as I was able
# to explore the correlation more and verify my variable importance to the instructions. With that, the final project
# method had a score for many other variables, while the in class method showed zero importance for all other 
# variable which does not seem likely. 

#---------------------------------------------------------------------------------------------------------------
# class taught method for tree models:

in_train = createDataPartition(y = house$price,
                               p = 0.8, list = FALSE)

house_price_train = house[in_train, ]
house_price_test = house[-in_train, ]


# Then, save the training set into a new dataframe called training_set. Here we can use the select function from the 
# dplyr package to select all the columns except price, which is the outcome we’re trying to predict.

training_set = select(house_price_train, -price)

tree_model = train(y = house_price_train$price,
                   x = training_set, 
                   method = "rpart")


tree_model

tree_model$finalModel

#visualize the tree

library(rpart.plot)

rpart.plot(tree_model$finalModel)

plot(varImp(tree_model))

#---------------------------------------------------------------------------------------------------------------
#         MODEL SELECTION
# Below is the code the final project says to use for model selection and testing. This also is much different from every lesson in class
# and the methods we learned. I will write the code here, but should this not produce desierable results, I will resort to the methods
# taught in class.

print(tree_model_2)

lm_model <- lm(price ~., data = house)
summary(lm_model)
confint(lm_model)

train_control <- trainControl(method = "cv", number = 3, savePredictions = TRUE)

model <- train(price ~ ., data = house, trControl = train_control, method = "lm")
model

train_control_2<- trainControl(method = "cv", number = 3, savePredictions = TRUE)

model_2 <- train(price ~ ., data = house, trControl = train_control, method = "rf")
model_2

# Prediction of Home Price

to_predict <- house[0,]
to_predict[1,]$sqft_living <- 4000
to_predict[1,]$sqft_above <- 4000
to_predict[1,]$sqft_lot <- 5000
to_predict[1,]$bedrooms <- 4
to_predict[1,]$bathrooms <- 3
to_predict[1,]$grade <- 7
to_predict[1,]$condition <- 5
to_predict[1,]$yr_built <- 2004
to_predict[1,]$yr_renovated <- 2004
to_predict[1,]$floors <- 2
to_predict[1,]$lat <- 47.5112
to_predict[1,]$long <- -122.257
to_predict[1,]$sqft_basement <- 0
to_predict[1,]$waterfront_fac <- 1
to_predict[1,]$view_fac <- 1



house_proc_steps = preProcess(house,
           method = c("center", "scale"))

house_proc = predict(house_proc_steps, newdata = house)


#summary(to_predict)
predictions_random_forest = predict(model_2, newdata = to_predict)
predictions_lm = predict(model, newdata = to_predict)

predictions_random_forest
predictions_lm

# the random forest model performs the best.


#------------------------------------------- Class Method for Model Selection and Prediction -------------------------------------------

house_2 = house

preprocess_steps = preProcess(house_2,
                              method = c("center", "scale"))

house_train_proc = predict(preprocess_steps, newdata = house_price_train)

house_test_proc = predict(preprocess_steps, newdata = house_price_test)

fit = train(price ~ ., data = house_train_proc,
            method = "lm", trControl = trainControl(method = 'cv',
                                                    number = 10), metric = 'RMSE')


fit
fit$finalModel
summary(fit)

fit_2 = train(price ~ grade + sqft_living + lat + sqft_above + long + bathrooms, 
              data = house_train_proc,
              method = "lm", 
              trControl = trainControl(method = 'cv',
                                       number = 10), 
              metric = 'RMSE')


# use predict() to make predictions

predictions = predict(fit, newdata = house_test_proc)

predictions_2 = predict(fit_2, newdata = house_test_proc)

# view the metric RMSE

postResample(pred = predictions, obs = house_test_proc$price)

postResample(pred = predictions_2, obs = house_test_proc$price)

#------------------------ Additional Exploration/Experimentation --------------------------------
# source: Kaggle online tutorial

library(tidyverse)
library(stringr)
library(lubridate)
library(DT)
library(caret)
library(leaflet)
library(corrplot)
library(boot) #for diagnostic plots

KCHouseData = house

rm(list=ls())

fillColor = "#FFA07A"
fillColor2 = "#F1C40F"


# peek into the data:
datatable(head(KCHouseData), style="bootstrap", class="table-condensed", options = list(dom = 'tp',scrollX = TRUE))

# the above is a strange way to look at the data, it shows 6 entries in the viewer section, with a scroll bar.

# bedrooms and price:
KCHouseData %>%
  group_by(bedrooms) %>%
  summarise(PriceMedian = median(price, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(bedrooms = reorder(bedrooms,PriceMedian)) %>%
  arrange(desc(PriceMedian)) %>%
  
  ggplot(aes(x = bedrooms,y = PriceMedian)) +
  geom_bar(stat='identity',colour="white", fill = fillColor) +
  geom_text(aes(x = bedrooms, y = 1, label = paste0("(",PriceMedian,")",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  labs(x = 'bedrooms', 
       y = 'Median Price', 
       title = 'bedrooms and Median Price') +
  coord_flip() + 
  theme_bw()


# sqft_living and price:
KCHouseData %>% 
  filter(!is.na(price)) %>% 
  filter(!is.na(sqft_living)) %>% 
  
  ggplot(aes(x=sqft_living,y=price))+
  geom_point(color = "blue")+
  
  stat_smooth(aes(x=sqft_living,y=price),method="lm", color="red")+
  theme_bw()+
  theme(axis.title = element_text(size=16),axis.text = element_text(size=14))+
  xlab("(Sqft Living)")+
  ylab("Price")


# map of house locations and price

KCHouseData$PriceBin<-cut(KCHouseData$price, c(0,250e3,500e3,750e3,1e6,2e6,999e6))

center_lon = median(KCHouseData$long,na.rm = TRUE)
center_lat = median(KCHouseData$lat,na.rm = TRUE)

factpal <- colorFactor(c("black","blue","yellow","orange","#0B5345","red"), 
                       KCHouseData$PriceBin)



leaflet(KCHouseData) %>% addProviderTiles("Esri.NatGeoWorldMap") %>%
  addCircles(lng = ~long, lat = ~lat, 
             color = ~factpal(PriceBin))  %>%
  # controls
  setView(lng=center_lon, lat=center_lat,zoom = 12) %>%
  
  addLegend("bottomright", pal = factpal, values = ~PriceBin,
            title = "House Price Distribution",
            opacity = 1)


# latitude vs. longitude
KCHouseData %>% 
  filter(!is.na(lat)) %>% 
  filter(!is.na(long)) %>% 
  
  ggplot(aes(x=lat,y=long))+
  geom_point(color = "blue")+
  
  theme_bw()+
  theme(axis.title = element_text(size=16),axis.text = element_text(size=14))+
  xlab("Latitude")+
  ylab("Longitude")




