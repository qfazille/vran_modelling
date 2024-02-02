library(dplyr)
library(pins)
library(tidymodels)
library(parsnip)
library(xgboost)
library(glue)
#library(kknn)

# 1- read data
bb <- board_folder(path = "00-pins")
x <- pin_read(board = bb, name = "ready_data") %>%
  select(-area_band)

x2 <- x[sample(1:nrow(x), size = 100000),]

tt <- x %>%
  group_by(code_postal) %>%
  summarise(vol = n())

# Specific trained_data that take into account the postcode we need to learn from
# Keep also the departments that are not that often
strate1 <- 2000
strate2 <- 1000
strate3 <- 200
strate4 <- 50

perc_strate0 <- 0.10 # for NA but by dept
perc_strate1 <- 0.08
perc_strate2 <- 0.15
perc_strate3 <- 0.25
perc_strate4 <- 0.5
perc_strate5 <- 1

x2 <- x %>%
  group_by(code_departement, code_postal) %>%
  mutate(vol = n()) %>%
  mutate(strate = case_when(
    is.na(code_postal) ~ "strate0",
    vol > strate1 ~ "strate1",
    vol > strate2 ~ "strate2",
    vol > strate3 ~ "strate3",
    vol > strate4 ~ "strate4",
    T ~ "strate5"
  )) %>%
  select(-vol) %>%
  as.data.frame()

train_data_strate0 <- training(initial_split(data = x2 %>% filter(strate == "strate0") %>% select(-strate), strata = "code_departement", prop = perc_strate1, pool = 0))
train_data_strate1 <- training(initial_split(data = x2 %>% filter(strate == "strate1") %>% select(-strate), strata = "code_postal", prop = perc_strate1, pool = 0.1))
train_data_strate2 <- training(initial_split(data = x2 %>% filter(strate == "strate2") %>% select(-strate), strata = "code_postal", prop = perc_strate2, pool = 0))
train_data_strate3 <- training(initial_split(data = x2 %>% filter(strate == "strate3") %>% select(-strate), strata = "code_postal", prop = perc_strate3, pool = 0))
train_data_strate4 <- training(initial_split(data = x2 %>% filter(strate == "strate4") %>% select(-strate), strata = "code_postal", prop = perc_strate4, pool = 0))
train_data_strate5 <- x2 %>% filter(strate == "strate5")

#train_data_strate0 <- x2 %>% filter(strate == "strate0") %>% select(-strate)
#train_data_strate1 <- training(initial_split(data = x2 %>% filter(strate == "strate1") %>% select(-strate), prop = 0.2))
#train_data_strate2 <- training(initial_split(data = x2 %>% filter(strate == "strate2") %>% select(-strate), prop = 0.05))

train_data <- train_data_strate0 %>%
  bind_rows(train_data_strate1) %>%
  bind_rows(train_data_strate2) %>%
  bind_rows(train_data_strate3) %>%
  bind_rows(train_data_strate4) %>%
  bind_rows(train_data_strate5) %>%
  select(-strate)

# To check
tt <- train_data %>% group_by(code_postal) %>% summarise(vol = n())
tt2 <- train_data %>% filter(is.na(code_postal)) %>% group_by(code_departement) %>% summarise(vol = n())

train_data1 <- train_data %>% filter(as.integer(code_departement) < 50) # j'aurai du faire as.integer(as.character()) tant pis
train_data1_cp <- train_data1 %>% pull(code_postal) %>% unique()
train_data1_cd <- train_data1 %>% pull(code_departement) %>% unique()
train_data2 <- train_data %>%
  filter(!code_postal %in% train_data1_cp)
train_data2_cd <- train_data2 %>% pull(code_departement) %>% unique()

###########################################
## Modeling using xgboost reg (part 1/2) ##
###########################################
{
  # Step 1: Create a recipe
  rec <- recipe(vran ~ ., data = train_data1) %>%
    step_dummy(code_departement, levels = unique(x$code_departement), one_hot = T) %>%
    step_dummy(type_local, levels = unique(x$type_local), one_hot = T) %>%
    step_dummy(code_postal, levels = unique(x$code_postal), one_hot = T) #%>%
    #step_dummy(c(area_band, type_local), one_hot = TRUE)
  
  # Step 2: Create an xgboost regression model specification
  xgboost_spec <- boost_tree(tree_depth = 2, 
                             trees = 1500,
                             learn_rate = 0.3) %>%
    set_engine("xgboost", objective = "reg:squarederror") %>%
    set_mode("regression")
  
  # Step 3: Create a workflow
  wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(xgboost_spec)
  
  
  # Step 4: Split data into training and testing sets
  #set.seed(123)
  #split <- initial_split(x, prop = 0.2)
  #train_data <- training(split)
  #test_data <- testing(split)
  
  # Step 5 : cross validation
  #my_validation_set <- vfold_cv(train_data, strata = vran, v = 10, repeats = 1)
  
  # Step 6: Fit the final model
  #final_wf <- fit_resamples(wf, resamples = my_validation_set)
  #collect_metrics(final_wf)
  
  final_fit1 <- wf %>%
    fit(data = train_data1)
  
  # Step 7: Make predictions on the test set
  to_test <- train_data[sample(1:nrow(train_data), size = 1000),]
  predictions <- predict(final_fit, new_data = to_test)
  
  to_test$pred <- round(predictions$.pred)
  
}

###########################################
## Modeling using xgboost reg (part 2/2) ##
###########################################
{
  # Step 1: Create a recipe
  rec <- recipe(vran ~ ., data = train_data2) %>%
    step_dummy(code_departement, levels = unique(x$code_departement), one_hot = T) %>%
    step_dummy(type_local, levels = unique(x$type_local), one_hot = T) %>%
    step_dummy(code_postal, levels = unique(x$code_postal), one_hot = T) #%>%
  #step_dummy(c(area_band, type_local), one_hot = TRUE)
  
  # Step 2: Create an xgboost regression model specification
  xgboost_spec <- boost_tree(tree_depth = 2, 
                             trees = 1500,
                             learn_rate = 0.3) %>%
    set_engine("xgboost", objective = "reg:squarederror") %>%
    set_mode("regression")
  
  # Step 3: Create a workflow
  wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(xgboost_spec)
  
  
  final_fit2 <- wf %>%
    fit(data = train_data2)
  
}

##########################
## The prediction table ##
##########################
{
  y <- pin_read(board = bb, name = "cleaned_data")
  pc <- read.csv("01-data/communes-departement-region.csv") %>%
    select(code_departement, code_postal) %>%
    distinct(code_postal, .keep_all = T)
  
  # To predict
  tp <- expand.grid(
    type_local = unique(y$type_local),
    code_postal = pc$code_postal
    #,area_band = unique(y$area_band)
  ) %>%
    left_join(pc, by = "code_postal") %>%
    select(code_departement, code_postal, type_local) %>% #, area_band) %>%
    mutate(code_departement = factor(code_departement),
           code_postal = factor(code_postal),
           type_local = factor(type_local))
           #,area_band = factor(area_band))
  
  # Predict the table
  tp1 <- tp %>% filter(code_departement %in% train_data1_cd)
  tp2 <- tp %>% filter(code_departement %in% train_data2_cd)
  
  preds1 <- predict(final_fit1, new_data = tp1)
  preds1 <- predict(final_fit2, new_data = tp2)
  
  tp1$vran_pred <- round(preds1$.pred)
  
  # Write to excel
  capitaux_predictions_file <- glue("02-outputs/capitaux_predictions_{format(Sys.Date(), '%Y%m%d')}.xlsx")
  openxlsx::write.xlsx(x = tp, file = capitaux_predictions_file)
  pin_write(board = bb, x = tp, name = "predictions")
}
