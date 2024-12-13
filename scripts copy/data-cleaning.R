library(tidyverse)
library(tidymodels)
library(reshape2)
library(ggraph)
library(httr)
library(igraph)
library(data.table)
library(ggplot2)
library(ggrepel)
library(plotly)
library(umap)
library(tm)
library(kableExtra)

# import data
df <- read.csv('./data/user_behavior_dataset.csv')
new_df <- df %>% 
  select(-User.ID)

# explore possible clusters
# Cluster 0: May represent users with high app usage and data consumption.
new_df %>% 
  ggplot(aes(x = App.Usage.Time..min.day., y = Data.Usage..MB.day., color = Operating.System)) + geom_point(alpha = .25) + labs(x = 'App Usage(Mins per Day)', y = 'Data Usage(MB per Day)')

# Cluster 1: Could be users with low screen time and fewer apps installed.
new_df %>% 
  ggplot(aes(x = Screen.On.Time..hours.day., y = Number.of.Apps.Installed, color = Operating.System)) + geom_point(alpha = .25) + labs(x = 'Screen Time(Hours per Day)', y = 'Number of Apps Installed')

# Cluster 2: Possibly users with moderate usage and average battery drain.
new_df %>% 
  ggplot(aes(x = App.Usage.Time..min.day., y = Battery.Drain..mAh.day., color = Operating.System)) + geom_point(alpha = .25) + labs(x = 'App Usage(Mins per Day)', y = 'Average Battery Drain(mAh/Day)')

# Cluster 3: Might be heavy users with many apps but efficient battery usage.
new_df %>% 
  ggplot(aes(x = App.Usage.Time..min.day., y = Data.Usage..MB.day., color = Operating.System)) + geom_point(alpha = .25) + labs(x = 'App Usage(Mins per Day)', y = 'Data Usage(MB per Day)')

# Personally I'm interested in possible clusters b/w high screentime users vs. app usage, are there people who check their phone more often

rec <- recipe(User.Behavior.Class ~ ., data = new_df) %>% 
  step_scale(all_numeric_predictors()) %>% 
  step_center(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) 

rec_prep <- prep(rec)

df_encoded <- bake(rec_prep, new_data = new_df)
head(df_encoded)

# cluster 0 k-means

cluster_0_df <- df_encoded %>% 
  select(App.Usage.Time..min.day., Data.Usage..MB.day.)

cluster_0 <- kmeans(cluster_0_df, 3, 20)


