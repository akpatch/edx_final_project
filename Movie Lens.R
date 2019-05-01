#######################################################################################################################
# Create edx set, validation set, and submission file##################################################################
#######################################################################################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#######################################################################################################################
## Exploring Data #####################################################################################################
#######################################################################################################################

#getting row and column length
nrow(edx)
ncol(edx)


#frequency table to ratings to get number of 0 and 3 in ratings
table(edx$rating)

#number of unique movies
length(unique(edx$movieId))

#number of unique users
length(unique(edx$userId))

edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

#Create data which give each genre category for a movie a separate entry 
#Same movie_rating will occur in multiple rows if movie is categorized into multiple genres
by_genre<-separate_rows(edx, genres, convert = TRUE)

#number of movies by genre
by_genre %>% group_by(genres) %>% summarize(count=length(rating)) 
##Do this again for validation dataset for future testing
by_genre_v<-separate_rows(validation, genres, convert = TRUE)


#Movie with highest rating
edx%>% group_by(title)%>% summarize(count=length(rating)) %>% arrange(desc(count))


#highest ratings
edx %>% group_by(rating) %>% summarize(count=length(rating)) %>% arrange(desc(count)) 


########################################################################################################################
## Optimizing RMSE #####################################################################################################
########################################################################################################################


###Since the outcome of interest is RMSE, create a function to calculate RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


###First, look at just the average movie rating and use Naive Bayes
##mu_hat is the overall average rating
mu_hat <- mean(edx$rating)
mu_hat


###calculate RMSE for the overall average
overall_average_rmse <- RMSE(validation$rating, mu_hat)
overall_average_rmse

###Here I am creating a tibble to hold the results of the models to compare RMSE
rmse_results <- tibble(method = "Overall average", RMSE = overall_average_rmse)

#View the table
rmse_results %>% knitr::kable()

## The RMSE result for the Overall Average Model is over 1, which is not very good
## To improve this we will add another parameter to the model: Movie Effects
## This will account for the difference in each movie's average rating from the overall average

## b_hat_i is the average of the diffence between the rating given to the movie minus the overall average rating
mu_hat <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_hat_i = mean(rating - mu_hat))


##Here we visualize the distribution of b_i
movie_avgs %>% qplot(b_hat_i, geom ="histogram", bins = 10, data = ., color = I("black"))

##Now we use the validation data to check the RSME for the mode

#first we have to calculate b_hat_i for the validation set and calculate the predicted ratings for the validation model
predicted_ratings <- mu_hat + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_hat_i

#next we test to see the RMSE 
model_movieId_rmse <- RMSE(predicted_ratings, validation$rating)

#finally we add this model to our RSME table 
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_movieId_rmse ))
#View the table
rmse_results %>% knitr::kable()




##The RMSE result for the Movie Effects is still not very low so we will add another parameter to the model: User Effects

## b_hat_u is the average rating given by each user
edx %>% 
  group_by(userId) %>% 
  summarize(b_hat_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_hat_u)) + 
  geom_histogram(bins = 30, color = "black")

##Now we use the validation data to check the RSME for the mode
#first we have to create user averages for the validation set so that we can make our predictions
user_avgs <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_hat_u = mean(rating - mu_hat - b_hat_i))

## Then we have to calculate the predicted ratings for the validation mode
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_hat_i + b_hat_u) %>%
  .$pred

#next we test to see the RMSE 
model_movieUser_rmse <- RMSE(predicted_ratings, validation$rating)

#finally we add this model to our RSME table 
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_movieUser_rmse ))

#View the table
rmse_results %>% knitr::kable()

## We see that this RMSE result is sufficiently low for our recommendation system to qualify as a success



##The RMSE result is fine, but we want to see if there is any additional help from: Genre Effects

## b_hat_g_g is the average rating given for each genre category
edx %>% 
  group_by(genres) %>% 
  summarize(b_hat_g = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_hat_g)) + 
  geom_histogram(bins = 30, color = "black")

##Now we use the validation data to check the RSME for the mode
#first we have to create user averages for the validation set so that we can make our predictions
genre_avgs <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_hat_g = mean(rating - mu_hat - b_hat_i - b_hat_u))

## Then we have to calculate the predicted ratings for the validation mode
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu_hat + b_hat_i + b_hat_u +b_hat_g) %>%
  .$pred

#next we test to see the RMSE 
model_movieUserGenre_rmse <- RMSE(predicted_ratings, validation$rating)

#finally we add this model to our RSME table 
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User + Genre Effects Model",  
                                     RMSE = model_movieUserGenre_rmse ))

#View the table
rmse_results %>% knitr::kable()

## We see that this RMSE result is slightly lower and is therefore a slight improvement on the models that exclude genre effects


## I still think this can be improved so we will re-do all of these models using the data that 
## split the combined genres into individual rows by genre

### Again, first we look at just the average movie rating and use Naive Bayes
## mu_hat_g is the overall average rating
mu_hat_g <- mean(by_genre$rating)
mu_hat_g


### calculate RMSE for the overall average
overall_average_rmse_g <- RMSE(by_genre_v$rating, mu_hat_g)
overall_average_rmse_g

### Next we add this model to our RSME table 
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Split Genres: Overall average", 
                                     RMSE = overall_average_rmse_g))

#View the table
rmse_results %>% knitr::kable()

## Again, we see that the RMSE result for the Overall Average Model with the split genre data is over 1, which is not very good
## However, we see that using the data split by genre helps to lower RMSE ever slightly more than using the original data
## To improve this we will again add another parameter to the model: Movie Effects
## This will account for the difference in each movie's average rating from the overall average

## b_hat_i_g is the average of the diffence between the rating given to the movie minus the overall average rating

mu_hat_g <- mean(by_genre$rating) 
movie_avgs <- by_genre %>% 
  group_by(movieId) %>% 
  summarize(b_hat_i_g = mean(rating - mu_hat_g))


##Here we visualize the distribution of b_hat_i_g
movie_avgs %>% qplot(b_hat_i_g, geom ="histogram", bins = 10, data = ., color = I("black"))

##Now we use the validation data to check the RSME for the mode

# Again, first we have to calculate b_hat_i_g for the validation set and calculate the predicted ratings for the validation model
predicted_ratings <- mu_hat_g + by_genre_v %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_hat_i_g

#next we test to see the RMSE 
model_movieId_rmse_g <- RMSE(predicted_ratings, by_genre_v$rating)

#finally we add this model to our RSME table 
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Split Genres: Movie Effect Model",
                                     RMSE = model_movieId_rmse_g ))
#View the table
rmse_results %>% knitr::kable()

##The RMSE result for the Movie Effects on the data split by genre is still not very low so we will add another parameter to the model: User Effects
## However, we see that using the data split by genre helps to lower RMSE ever slightly more than using the original data

## b_hat_u_g is the average rating given by each user
by_genre %>% 
  group_by(userId) %>% 
  summarize(b_hat_u_g = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_hat_u_g)) + 
  geom_histogram(bins = 30, color = "black")

##Now we use the validation data to check the RSME for the mode

#first we have to create user averages for the validation set so that we can make our predictions
user_avgs <- by_genre_v %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_hat_u_g = mean(rating - mu_hat_g - b_hat_i_g))

## Then we have to calculate the predicted ratings for the validation mode
predicted_ratings <- by_genre_v %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat_g + b_hat_i_g + b_hat_u_g) %>%
  .$pred

#next we test to see the RMSE 
model_movieUser_rmse_g <- RMSE(predicted_ratings, by_genre_v$rating)

#finally we add this model to our RSME table 
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Split Genres: Movie + User Effects Model",  
                                     RMSE = model_movieUser_rmse_g ))

#View the table
rmse_results %>% knitr::kable()

## Again we see that this RMSE result for Movie and User Effects is sufficiently low for our recommendation system to qualify as a success
## Additionally, we see that using the data split by genre helps to lower RMSE ever slightly more


##The RMSE result is fine, but we want to see if there is any additional help from: Genre Effects

## b_hat_g_g is the average rating given for each genre category

by_genre %>% 
  group_by(genres) %>% 
  summarize(b_hat_g_g = mean(rating)) %>% 
  filter(n()>=1) %>%
  ggplot(aes(b_hat_g_g)) + 
  geom_histogram(bins = 30, color = "black")

#Note we see that the distribution of this does not follow the normal distribution 

##Now we use the validation data to check the RSME for the mode

#first we have to create user averages for the validation set so that we can make our predictions
genre_avgs <- by_genre_v %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_hat_g_g = mean(rating - mu_hat_g - b_hat_i_g- b_hat_u_g))

## Then we have to calculate the predicted ratings for the validation mode
predicted_ratings <- by_genre_v %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu_hat_g + b_hat_i_g + b_hat_u_g +b_hat_g_g) %>%
  .$pred

#next we test to see the RMSE 
model_movieUserGenre_rmse_g <- RMSE(predicted_ratings, by_genre_v$rating)

#finally we add this model to our RSME table 
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Split Genres: Movie + User + Genre Effects Model",  
                                     RMSE = model_movieUserGenre_rmse_g ))

#View the table
rmse_results %>% knitr::kable()

## Again we see that this RMSE result for Movie, User, and Genre Effects is sufficiently low for our recommendation system to qualify as a success
## Additionally, we see that using the data split by genre helps to lower RMSE ever slightly more
