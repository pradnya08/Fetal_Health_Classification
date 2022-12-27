#
# Creator:        Andrew Disher, RutviRahul Gadre, Pradnya Jagadish Asundi
# Affiliation:    UMASS Dartmouth
# Course:         CIS 550 - 001
# Assignment:     Final Project
# Date:           11/2/2022
#
# TASK: Fit a multinomial logistic regression model and perform diagnostics on the Fetal Health Data Set. Then 
#       evaluate the performance of the final model. 
#
# Data Source: 
#
# Fetal Health Data Set
# https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

# Packages
library(car) # For cumsum() function; computes the cumulative sum of a vector
library(caret) # For Confusion matrix creation
library(pROC) # For creating an ROC curve and calculating Area Under Curve
library(nnet) # For function to fit a multinomial logistic regression model
library(ROSE) # For over/under sampling of the data


# -------------------------------------------------------------------------
# Import the data ---------------------------------------------------------
# -------------------------------------------------------------------------

Fetal_Health_Data_Set <- read.csv("C:/Users/pradn/OneDrive/Desktop/Data Science/AML/Project 2/fetal_health.csv")


# Split Data into Training and Testing in R 
sample_size = floor(0.8*nrow(Fetal_Health_Data_Set))
set.seed(666)

# Randomly split data
picked = sample(seq_len(nrow(Fetal_Health_Data_Set)), size = sample_size)

# Store the Training and Testing data in their respective data frames
Training_Data <- Fetal_Health_Data_Set[picked, ]
Test_Data <- Fetal_Health_Data_Set[-picked, ]

# Obtain proportions for response variable classes
prop.table(table(Training_Data$fetal_health))


# -------------------------------------------------------------------------
# Correct imbalanced class issue ------------------------------------------
# -------------------------------------------------------------------------

# NOTE: To correct for class imbalance, we are going to create two subsets that we can then 
#       run the ROSE:ovun.sample() function on to produce a corrected version of the data. 
#       Then, we will bind both of the balanced subsets together to obtain a single balanced
#       data set. 

# Create subsets of data (each have class 1, and then either 2 or 3)
subset12 <- base::subset(Training_Data, fetal_health == 1 | fetal_health == 2)

subset13 <- base::subset(Training_Data, fetal_health == 1 | fetal_health == 3)



# Use a combination of over and under sampling to acquire more balanced data sets

# Subset12
Training_Data12 <- ovun.sample(fetal_health~., data=subset12, method = "both",
                             p = 0.47, # Probability of resampling from the rare class (has heart disease)
                             seed = 666,
                             N = nrow(subset12))$data

# Subset13
Training_Data13 <- ovun.sample(fetal_health~., data=subset13, method = "both",
                               p = 0.47, # Probability of resampling from the rare class (has heart disease)
                               seed = 666,
                               N = nrow(subset13))$data

# Check the distribution of the response variable in the new data sets. 
prop.table(table(Training_Data12$fetal_health))
prop.table(table(Training_Data13$fetal_health))

# Since we included class = 1 in both subsets, we must make sure we don't include it's data 
# twice when we bind the two subsets together.
Training_Data_Balanced <- rbind(Training_Data12[Training_Data12$fetal_health == 2,], Training_Data13)

prop.table(table(Training_Data_Balanced$fetal_health))

# -------------------------------------------------------------------------
# Create a preliminary logistic regression model using all predictors -----
# -------------------------------------------------------------------------

Model1 <- multinom(fetal_health ~., data = Training_Data_Balanced)
summary(Model1)


# Create a data frame to store information about the model (coefficients, std errors, p-values)
Model1Summary <- as.data.frame(summary(Model1)$coefficients)

# Adjust columns in Model1Summary
Model1Summary <- t(Model1Summary)
Model1Summary <- data.frame(rownames(Model1Summary), Model1Summary, summary(Model1)$standard.errors[1,], summary(Model1)$standard.errors[2,])
# Model1Summary <- Model1Summary[-1, ]
row.names(Model1Summary) <- NULL
colnames(Model1Summary) <- c("Parameter", "Coef. Estimate 2", "Coef. Estimate 3", "Std. Error 2", "Std. Error 3")


# -------------------------------------------------------------------------
# Hyp. Tests for Equation Comparing class 1 to 2 --------------------------
# -------------------------------------------------------------------------

Model1Summary12 <- Model1Summary[, c(1,2,4)]

# Acquire Z-scores
Model1Summary12$Z <- Model1Summary12$`Coef. Estimate 2`/Model1Summary12$`Std. Error 2`

# Acquire p-values (this is a two-tailed Wald test, with beta = 0 as null hypothesis)
Model1Summary12$`P Value` <- (1 - pnorm(abs(Model1Summary12$Z), 0, 1)) * 2

# Create significance column
Model1Summary12$Significant <- vector("character", nrow(Model1Summary12))

for (row in 1:nrow(Model1Summary12)) {
  if(Model1Summary12[row, 5] <= .001){
    Model1Summary12[row, 6] = '***'
  }else if(Model1Summary12[row, 5] <= .01){Model1Summary12[row, 6] = '**'}
  else if(Model1Summary12[row, 5] <= .05){Model1Summary12[row, 6] = '*'}
  else if(Model1Summary12[row, 5] > .05){Model1Summary12[row, 6] = 'Not Sig.'}
}
View(Model1Summary12)


# -------------------------------------------------------------------------
# Hyp. Tests for Equation Comparing class 1 to 3 --------------------------
# -------------------------------------------------------------------------

Model1Summary13 <- Model1Summary[, c(1,3,5)]

# Acquire Z-scores
Model1Summary13$Z <- Model1Summary13$`Coef. Estimate 3`/Model1Summary13$`Std. Error 3`

# Acquire p-values (this is a two-tailed Wald test, with beta = 0 as null hypothesis)
Model1Summary13$`P Value` <- (1 - pnorm(abs(Model1Summary13$Z), 0, 1)) * 2

# Create significance column
Model1Summary13$Significant <- vector("character", nrow(Model1Summary13))

for (row in 1:nrow(Model1Summary13)) {
  if(Model1Summary13[row, 5] <= .001){
    Model1Summary13[row, 6] = '***'
  }else if(Model1Summary13[row, 5] <= .01){Model1Summary13[row, 6] = '**'}
  else if(Model1Summary13[row, 5] <= .05){Model1Summary13[row, 6] = '*'}
  else if(Model1Summary13[row, 5] > .05){Model1Summary13[row, 6] = 'Not Sig.'}
}
View(Model1Summary13)


# NOTE: We have calculated p-values for each of the coefficients estimates. Some of the coefficients are NOT significant
#       in both equations, whereas there is some disagreement regarding other variables. We will keep the variables that 
#       appear to be significant in one equation and not in the other, whereas we will remove variables that are NOT 
#       significant in either equation, and fit a new model, done in the next section below.  


# -------------------------------------------------------------------------
# Model 2 Fitting ---------------------------------------------------------
# -------------------------------------------------------------------------

Model2 <- multinom(fetal_health ~., data = subset(Training_Data_Balanced, select = -c(histogram_tendency, histogram_number_of_zeroes, 
                                                                                      histogram_min, mean_value_of_long_term_variability)))
summary(Model2)


# Create a data frame to store information about the model (coefficients, std errors, p-values)
Model2Summary <- as.data.frame(summary(Model2)$coefficients)

# Adjust columns in Model2Summary
Model2Summary <- t(Model2Summary)
Model2Summary <- data.frame(rownames(Model2Summary), Model2Summary, summary(Model2)$standard.errors[1,], summary(Model2)$standard.errors[2,])
# Model2Summary <- Model2Summary[-1, ]
row.names(Model2Summary) <- NULL
colnames(Model2Summary) <- c("Parameter", "Coef. Estimate 2", "Coef. Estimate 3", "Std. Error 2", "Std. Error 3")


# -------------------------------------------------------------------------
# Hyp. Tests for Equation Comparing class 1 to 2 --------------------------
# -------------------------------------------------------------------------

Model2Summary12 <- Model2Summary[, c(1,2,4)]

# Acquire Z-scores
Model2Summary12$Z <- Model2Summary12$`Coef. Estimate 2`/Model2Summary12$`Std. Error 2`

# Acquire p-values
Model2Summary12$`P Value` <- (1 - pnorm(abs(Model2Summary12$Z), 0, 1)) * 2

# Create significance column
Model2Summary12$Significant <- vector("character", nrow(Model2Summary12))

for (row in 1:nrow(Model2Summary12)) {
  if(Model2Summary12[row, 5] <= .001){
    Model2Summary12[row, 6] = '***'
  }else if(Model2Summary12[row, 5] <= .01){Model2Summary12[row, 6] = '**'}
  else if(Model2Summary12[row, 5] <= .05){Model2Summary12[row, 6] = '*'}
  else if(Model2Summary12[row, 5] > .05){Model2Summary12[row, 6] = 'Not Sig.'}
}
View(Model2Summary12)


# -------------------------------------------------------------------------
# Hyp. Tests for Equation Comparing class 1 to 3 --------------------------
# -------------------------------------------------------------------------

Model2Summary13 <- Model2Summary[, c(1,3,5)]

# Acquire Z-scores
Model2Summary13$Z <- Model2Summary13$`Coef. Estimate 3`/Model2Summary13$`Std. Error 3`

# Acquire p-values
Model2Summary13$`P Value` <- (1 - pnorm(abs(Model2Summary13$Z), 0, 1)) * 2

# Create significance column
Model2Summary13$Significant <- vector("character", nrow(Model2Summary13))

for (row in 1:nrow(Model2Summary13)) {
  if(Model2Summary13[row, 5] <= .001){
    Model2Summary13[row, 6] = '***'
  }else if(Model2Summary13[row, 5] <= .01){Model2Summary13[row, 6] = '**'}
  else if(Model2Summary13[row, 5] <= .05){Model2Summary13[row, 6] = '*'}
  else if(Model2Summary13[row, 5] > .05){Model2Summary13[row, 6] = 'Not Sig.'}
}
View(Model2Summary13)


# -------------------------------------------------------------------------
# Model Evaluation Metrics ------------------------------------------------
# -------------------------------------------------------------------------

Test_Predictions <- predict(Model2, subset(Test_Data, select = -c(fetal_health, histogram_tendency, histogram_number_of_zeroes, 
                                                                  histogram_min, mean_value_of_long_term_variability)))

# Produce a confusion matrix for the predictions
Confusion_Matrix <- confusionMatrix(data = Test_Predictions, reference = as.factor(Test_Data$fetal_health))
Confusion_Matrix
Confusion_Matrix$byClass
Confusion_Matrix$table


# -------------------------------------------------------------------------
# Create ROC Curves  ------------------------------------------------------
# -------------------------------------------------------------------------

# Store ROC curve information
roc_score = multiclass.roc(Test_Data$fetal_health, 
                           predict(Model2, subset(Test_Data, select = -c(fetal_health, histogram_tendency, histogram_number_of_zeroes, 
                                                                         histogram_min, mean_value_of_long_term_variability)), 
                                   type = 'prob'))


# Store the first element of roc_score in object rs
rs <- roc_score[['rocs']]

# Compute the areas under the curves, individually
auc(rs[[1]][[2]]) # 0.9528
auc(rs[[2]][[2]]) # 0.9746
auc(rs[[3]][[2]]) # 0.9527

# Plot each of the curves, overlayed with one another
plot.roc(rs[[1]][[2]], col = 'blue', legacy.axes = TRUE,
         main = "Multiclass ROC Curve -- Logistic Regression -- Fetal Health")
plot.roc(rs[[2]][[2]], col = 'red', add = TRUE)
plot.roc(rs[[3]][[2]], col = 'green', add = TRUE)

# Add legends for clarity
legend(x = "topright",          # Position
       legend = c("C1~C2", "C1~C3", "C2~C3"),  # Legend texts
       lty = c(1, 1, 1),           # Line types
       col = c('blue', 'red', 'green'),           # Line colors
       lwd = 2) 

legend(x = "bottomright",          # Position
       legend = c("AUC = 0.9528", "AUC = 0.9746", "AUC = 0.9527"),  # Legend texts
       lty = c(1, 1, 1),           # Line types
       col = c('blue', 'red', 'green'),           # Line colors
       lwd = 2, 
       horiz = TRUE)


















