# Yearly Word Frequency Analysis

## Overview
This module performs yearly word frequency analysis on YouTube delivery robot comments from 2015 to 2023. It processes raw word frequency data by cleaning, filtering stop words, merging synonyms, and generating cleaned datasets for further analysis.

## Directory Structure
```
yearly_word_frequency/
├── code/
│   └── yearly_word_frequency.ipynb     # Jupyter notebook with analysis pipeline
├── Input/
│   └── word_frequency_by_year_2015_2023.xlsx  # Raw input data
└── output/
    ├── word_frequency_by_year_2015_2023_cleaned.xlsx  # Cleaned dataset
    └── word_frequency.xlsx                            # Consolidated results
```

## Input Data Format
The input Excel file contains yearly word frequency data with columns:
- `{year}_word`: Words appearing in comments for each year (2015-2023)
- `{year}_frequency`: Frequency counts for corresponding words

## Processing Steps
1. **Data Loading**: Load raw Excel data and convert frequency columns to integer type
2. **Data Cleaning**: Remove rows where all frequency columns are NaN
3. **Stop Word Removal**: Filter out irrelevant words including:
   - Common stop words (never, dont, cant, etc.)
   - Generic terms (one, thing, lol, etc.)
   - Low-value words (someone, anyone, think, going)
   - Special characters and noise terms
4. **Synonym Merging**: Consolidate synonyms to reduce redundancy:
   - robot → automaton, android, cyborg
   - people → individuals, persons, humans
   - steal → pilfer, thieve, swipe
   - wonderful → marvelous, remarkable, magnificent
   - nice → pleasant, agreeable, delightful
   - cute → adorable, charming, endearing
5. **Output Generation**: Export cleaned data to Excel files

## Output Files
- `word_frequency_by_year_2015_2023_cleaned.xlsx`: Cleaned dataset with filtered words and merged synonyms
- `word_frequency.xlsx`: Consolidated word frequency analysis results

## Usage
1. Ensure input data is placed in the `Input/` directory
2. Open and run `code/yearly_word_frequency.ipynb` in Jupyter notebook
3. Processed files will be generated in the `output/` directory

## Dependencies
- pandas
- nltk (WordNet for synonym handling)
- openpyxl (for Excel file operations)

## Integration with YouTube-SC Project
This module is part of the larger YouTube-SC (YouTube Sentiment and Clustering Analysis) project, providing cleaned word frequency data for sentiment classification, clustering analysis, and topic modeling modules.