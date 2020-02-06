#Package
library(dplyr)
library(ggplot2)
library(MASS)
library(tm)
library(topicmodels)
library(tidytext)
library(caret)
library(glmnet)
library(readr)
library(ISLR)
library(stringr)
library(gam)
library(dbarts)
library(rpart)

set.seed(12345)

#HW1 Matrix

matrix(data = 1:6, nrow = 2, ncol = 3)
#Dataframe
data.frame("name"= c("A","B","C","D"),
           "Value" = c(1,2,3,4))
#The main difference is that you can expect only one and same data type in matrix; 
#however, in data frame, different data types can be expected. 
#Data frame also has column name and row names, while matrix does not.


#HW2 Matrix Algebra, dplyr, ggplot(week 4)
A <- matrix(c(5, 6, 1, 2, 2, 3), nrow = 2, ncol = 3)
B <- matrix(c(3, -2, 4, -3, 5, 6), nrow = 2, ncol = 3)
#B %*% A 
#A and y are said to be orthogonal if x^Ty = 0
#(xy)^-1 = y^-1 x^-1
#(x^T)^-1 = (x^-1)^T
#X=UDV⊤   X⊤X = VD2V⊤  
#V is matrix of eigenvectors D is eigenvalues
#K_mean and Hierarchical Clustering (WEEK 6)

###select mutate filter arrange 
counties %>%
  distinct(select(state, county, population, private_work, public_work, self_employed)) %>%
  mutate( public_workers = public_work * population / 100) %>%
  filter(state == "California" & population > 1000000)%>%
  arrange(desc(population))
#count
counties_selected %>%
  count(region,sort=TRUE)
counties_selected %>%
  count(state, wt= citizens, sort = TRUE)
#group_by, summarize
counties_selected %>%
  group_by(state) %>%
  summarize(total_area = sum(land_area),
            total_population = sum(population), n = n()) %>%
  mutate(density = total_population / total_area)%>%
  arrange(desc(density))
#ggplot
ggplot(apts) + 
  geom_point(aes(x = area, y = totalprice / 1000, 
                 col = rooms, shape = toilets, 
                 size = storage)) + 
  geom_smooth(aes(x = area, y = totalprice / 1000), colour = "red", se = FALSE) +
  xlab("Area (square meters)") + ylab("Price (in 1,000 Euros)")

ggplot(chickwts) + geom_boxplot(aes(x = feed, y = weight))
ggplot(Cars93) + geom_histogram(aes(x = Price)) + facet_wrap(~ DriveTrain)
ggplot(Auto) + geom_boxplot(aes(x = "", y = weight)) + facet_wrap(~ cylinders)



#HW3 TEXT MINING
#DTM
corpus <- Corpus(DirSource(directory = "Speeches_May_1967"))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, stripWhitespace) 
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)
dtm <- DocumentTermMatrix(corpus)
output <- LDA(dtm, k = 3, control = list(seed = 12345L))
beta <- tidy(output, matrix = "beta")
filter(beta, topic == 1) %>% arrange(desc(beta))

#join
users <- mutate(users, user_id = id)
combined <- left_join(tweets, users)
#VS
corpus_raw <- Corpus(VectorSource(combined$hashtags))
corpus <- tm_map(corpus_raw, content_transformer(tolower))
corpus <- tm_map(corpus, FUN = function(x) gsub(",", " ", x, fixed = TRUE))
corpus <- tm_map(corpus, stemDocument)
dtm <- DocumentTermMatrix(corpus)
corpus_tidy <- tidy(dtm)
corpus_tidy %>% bind_tf_idf(term, document, count) %>% arrange(desc(tf_idf))

tidy_con <- constitution %>%
  unnest_tokens(word, preamble)%>%
  anti_join(stop_words) %>%
  count(country_year, word) %>%
  cast_dtm(country_year, word, n)

#hierarchical clustering
hc_single <- hclust(dist(dtm), method = "single")
plot(hc_single, main = "Single Linkage", xlab = "", sub = "")
head(cutree(hc_single, 5))

#HW4 MODEL

ROOT <- "https://archive.ics.uci.edu/ml/machine-learning-databases/"
crime <- read.csv(paste0(ROOT, "communities/communities.data"),
                  header = FALSE, na.strings = "?")
colnames(crime) <- read.table(paste0(ROOT, "communities/communities.names"),
                              skip = 75, nrows = ncol(crime))[,2]
#check na
colSums(is.na(crime))
#remove row with na.
payback <- na.omit(payback)
crime <- crime[ , apply(crime, 2, function(x) !any(is.na(x)))]
crime$communityname <- NULL
#repalce NA with mean
crime$OtherPerCap[is.na(crime$OtherPerCap)] <- mean(crime$OtherPerCap, na.rm = TRUE)
#split train and test
in_train <- createDataPartition(y = crime$population, p = 0.8, list = FALSE)
training1 <- crime[ in_train, ]
testing1  <- crime[-in_train, ]

#step
mod1_AIC <- step(mod1_train, trace = FALSE)
Yhat_mod1AIC <- predict(mod1_AIC, newdata = testing)
defaultSummary(data.frame(obs = testing$Outstate, pred = Yhat_mod1AIC))

#fit glmnet
ctrl <- trainControl(method = "cv", number = 10)
fitglm <- train(ViolentCrimesPerPop ~ ., data = training1, 
                method = "glmnet", trControl = ctrl)
y_hat1 <- predict(fitglm, newdata = testing1)
defaultSummary(data.frame(obs = testing1$ViolentCrimesPerPop, pred = y_hat1))
#fit pls
pls_grid <- data.frame(.ncomp = 1:100)
PLS <- train(ViolentCrimesPerPop ~ (.)^2, data = training, method = "pls",
             trControl = ctrl, tuneGrid = pls_grid, preProcess = c("center", "scale"))
y_hat <- predict(PLS, newdata = testing)
defaultSummary(data.frame(obs = testing$ViolentCrimesPerPop, pred = y_hat))
#if y is factor
loans$y <- factor(loans$y, labels = c("yes", "no"), levels = 1:0)
LDA0 <- train(y ~Amount.Requested + Debt.To.Income.Ratio 
              + Employment.Length, data = training, 
              method = "lda",
              preProcess = c("center", "scale"))
sum_LDA0 <- confusionMatrix(predict(LDA0, newdata = testing), reference = testing$y)
round(sum_LDA0$overall,5)


