#########################################################################################
# IMPORT LIBRARIES
#########################################################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(dplyr, warn.conflicts = FALSE)


#########################################################################################
# IMPORT DATASET
#########################################################################################

dl <- as.data.frame(fread(file = "migr_asydcfsta.tsv", sep = "\t", header = TRUE, nrows = 500))

labels <- separate(data = dl, col= 1, into = c("unit","citizen","sex","age","decision","geo"), sep = ",")
