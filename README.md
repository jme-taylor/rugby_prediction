# Rugby Prediction

This repo contains my attempts to build a machine learning model that can predict the outcome of a rugby match. This is intended as a demonstration in my skills in developing, maintaining and productionising a machine learning model.

It also contains the baseline model setting, exploratory analysis and model training/parameter optimisation. This is so that I can illustrate my approach to training the model. In a production environment, these wouldn't be present in the model repository, but stored in version control in a seperate repository.

## Installation

Firstly, clone the repository.

I've used [poetry](https://python-poetry.org/) for dependecy management and packaging. If you'd like to replicate this project yourself, first make sure you've got poetry installed ([instructions](https://python-poetry.org/docs/#installation) if you haven't).

Once you have the repository cloned and poetry is installed, use your terminal to navigate to the local version of this repository and run the following commands.

```bash
poetry install
poetry shell
```

## Data

All the data is scraped with my custom built [rugby stats scraper](https://github.com/jme-taylor/rugby_stats_scraper). I haven't uploaded any of the data to this specific repository, but if you'd like to replicate this project you can use the scraper to get the same output as raw data I use here.
