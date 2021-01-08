# kaggle_projects
All the projects that have been submitted or drafted for the kaggle competitions.

kaggle.com

https://www.kaggle.com/rafamikuowski 

## House Prices: Advanced Regression Techinques

https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Project to start with kaggle, which scope is to predict the price of the housing in Boston.

Practisting different regression techniques, applying feature engineering ideas and HPO.

## Rock, paper, scissors

https://www.kaggle.com/c/rock-paper-scissors

Project to practise optimization techniques, where the bots are competing against each other in rock, paper, scissors game.
Each bot submitted plays with another one 1000 times, based on which it is concluded which one is better.

Good opportunity to practise different optimization techniques, from multi-armed bandit to reinforcement learning such as Q-learning. 

# Getting data using kaggle API 
Following https://github.com/Kaggle/kaggle-api

1. install kaggle using 'pip install kaggle'
2. get token from your account on kaggle.com (https://www.kaggle.com/<username>/account), save it in Downloads
3. go to the directory, which unables to move the file from Downloads to .kaggle
4. mv (directory_with_kaggle.json - following previous points it would be Downloads) (/.kaggle)
5. chmod 600 /XXX/.kaggle/kaggle.json
6. run 'kaggle competitions list' to see, if connection works
7. switch to directory, in which project will be developped
8. kaggle competitions download -c (your_competition_name
9. unzip (your_zip_file.zip)
