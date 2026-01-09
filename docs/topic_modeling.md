# Topic Modeling

This document describes the topic modeling implementation in the YouTube-SC project, which uses Latent Dirichlet Allocation (LDA) to discover discussion themes in YouTube comments about sidewalk delivery robots.

## Overview
Topic modeling is applied to identify and analyze the main themes and concerns expressed in public comments about delivery robots. The implementation uses LDA to extract latent topics from the comment corpus, providing insights into public perceptions beyond sentiment analysis.

## Method: Latent Dirichlet Allocation (LDA)

### Algorithm Description
LDA is a generative probabilistic model that assumes each document is a mixture of topics, and each topic is a distribution over words. The algorithm discovers these latent topic structures from the observed document-term matrix.

### Mathematical Formulation
- **Documents**: D = {d₁, d₂, ..., dₙ} where each document is a YouTube comment
- **Topics**: T = {t₁, t₂, ..., tₖ} where k is the number of topics
- **Words**: W = {w₁, w₂, ..., wᵥ} from the vocabulary
- **Parameters**:
  - α: Dirichlet prior for document-topic distributions
  - β: Dirichlet prior for topic-word distributions

### Inference Method
- **Gibbs Sampling**: Markov Chain Monte Carlo method for approximate inference
- **Iterations**: 1000-2000 passes for convergence
- **Burn-in**: First 100-200 samples discarded
- **Thinning**: Keep every 10th sample to reduce autocorrelation

## Configuration

### Topic Count Selection
- **Initial range**: 5-20 topics tested
- **Optimal value**: 10 topics based on coherence scores
- **Validation**: Multiple runs with different random seeds

### Preprocessing for Topic Modeling

#### Text Preparation
1. **Tokenization**: Split comments into individual words
2. **Stopword Removal**:
   - Standard English stopwords (NLTK list)
   - Custom stopwords specific to delivery robot domain
   - High-frequency but low-information terms
3. **Bigram Detection**: Identify common two-word phrases using gensim's Phrases model
4. **Lemmatization**: Reduce words to base forms using WordNet

#### Vocabulary Filtering
- **Minimum frequency**: Remove terms appearing in < 5 documents
- **Maximum frequency**: Remove terms appearing in > 50% of documents
- **Document frequency**: Keep terms with moderate document frequency for discriminative power

### Model Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| **num_topics** | 10 | Number of latent topics to extract |
| **chunksize** | 2000 | Number of documents processed at once |
| **passes** | 10 | Number of passes through corpus during training |
| **iterations** | 1000 | Number of sampling iterations |
| **alpha** | 'auto' | Let model learn asymmetric document-topic prior |
| **eta** | 'auto' | Let model learn asymmetric topic-word prior |
| **random_state** | 42 | Fixed seed for reproducible results |
| **per_word_topics** | True | Compute per-word topic distributions |

## Model Evaluation

### Coherence Measures

#### C_v Coherence
- Measures semantic similarity between high-probability words in topics
- Higher values indicate more interpretable topics
- Used for model selection and topic count optimization

#### UMass Coherence
- Based on document co-occurrence statistics
- More computationally efficient
- Used for monitoring convergence during training

#### Perplexity
- Measures how well the model predicts held-out documents
- Lower values indicate better generalization
- Calculated on validation set

### Human Evaluation

#### Topic Interpretability
- Manual labeling of topics by domain experts
- Assessment of topic consistency and clarity
- Inter-rater agreement calculation

#### Topic Stability
- Multiple runs with different random seeds
- Jaccard similarity between top words across runs
- Stability analysis to ensure robust topic extraction

## Interpretation Framework

### Keyword Analysis

#### Top-N Words per Topic
- Extract 10-15 highest probability words for each topic
- Calculate word salience (distinctiveness from other topics)
- Visualize using word clouds and bar charts

#### Topic-Document Distributions
- For each document, distribution over topics
- Identify dominant topics for each comment
- Aggregate topic proportions across the corpus

### Thematic Categories

#### Manual Labeling Process
1. **Keyword Review**: Examine top words for each topic
2. **Context Analysis**: Review representative documents for each topic
3. **Label Assignment**: Assign descriptive labels capturing topic essence
4. **Category Grouping**: Group related topics into broader themes

#### Example Topics from SDR Analysis
Based on the paper findings, topics included:
1. **Job Security Concerns**: Keywords: jobs, employment, workers, replace, automation
2. **Pedestrian Safety**: Keywords: sidewalk, safety, pedestrians, collision, hazard
3. **Robot Security**: Keywords: theft, vandalism, security, protect, camera
4. **Technology Adoption**: Keywords: future, technology, innovation, progress, smart
5. **Public Acceptance**: Keywords: community, acceptance, residents, neighborhood

### Visualization

#### PyLDAVis
- Interactive visualization of topic distances and term distributions
- Shows inter-topic relationships and term relevance
- HTML output for exploration and reporting

#### Topic Evolution
- Analyze topic prevalence over time (if temporal data available)
- Track changes in public concerns across different years
- Visualize using stacked area charts or heatmaps

## Implementation Details

### Code Structure
- **Main script**: `topic_modeling/topic_modeling_analysis.py`
- **Configuration**: `topic_modeling/config.py` for parameter management
- **Utilities**: Helper functions for preprocessing, visualization, and evaluation

### Workflow
1. **Data Loading**: Load preprocessed comments from dataset
2. **Corpus Creation**: Create dictionary and document-term matrix
3. **Model Training**: Train LDA model with specified parameters
4. **Evaluation**: Calculate coherence scores and perplexity
5. **Interpretation**: Extract topics, assign labels, analyze distributions
6. **Visualization**: Generate plots and interactive visualizations
7. **Reporting**: Save results to `results/` directory

### Output Files
- **Model files**: `.ldamodel` (gensim model), `.dict` (dictionary)
- **Topic summaries**: CSV with top words, coherence scores, topic labels
- **Document-topic distributions**: CSV mapping comments to topic probabilities
- **Visualizations**: PNG/HTML files for topic visualizations
- **Reports**: Summary statistics and interpretation notes

## Integration with Other Analyses

### Relationship to Sentiment Analysis
- **Sentiment-Topic Correlation**: Analyze which topics correlate with positive/negative sentiment
- **Topic-Specific Sentiment**: Calculate sentiment distributions within each topic
- **Joint Analysis**: Combine topic and sentiment for nuanced understanding

### Connection to Clustering
- **Complementary Approaches**: Topics provide thematic labels, clustering finds natural groups
- **Validation**: Compare topic assignments with cluster assignments
- **Hybrid Analysis**: Use topic distributions as features for clustering

### Text Statistics Integration
- **Topic Frequency**: Track prevalence of topics over time
- **Vocabulary Analysis**: Relate topic keywords to overall word frequencies
- **Statistical Testing**: Test differences in topic proportions across subgroups

## Applications to SDR Research

### Research Questions Addressed
1. **What are the main concerns** expressed about sidewalk delivery robots?
2. **How do discussions evolve** over time as technology matures?
3. **What thematic differences exist** between positive and negative comments?
4. **How do public perceptions align** with industry and policy discussions?

### Policy Implications
- **Safety concerns**: Inform safety regulations and robot design
- **Public acceptance**: Guide community engagement strategies
- **Employment impacts**: Address job displacement concerns
- **Technology adoption**: Understand barriers to widespread use

## Best Practices

### Parameter Tuning
- Use coherence scores rather than perplexity for model selection
- Test multiple random seeds to ensure topic stability
- Consider asymmetric priors (α, η) for more realistic topic distributions

### Interpretation Guidelines
- Consider context beyond just top words
- Look at representative documents for each topic
- Be cautious about over-interpreting minor topics
- Validate findings with domain knowledge

### Reproducibility
- Fix random seeds for reproducible topic assignments
- Save all model parameters with outputs
- Version control data and code for exact replication

## Limitations and Future Work

### Current Limitations
- **Bag-of-words assumption**: LDA ignores word order and syntax
- **Static analysis**: Basic LDA doesn't capture topic evolution over time
- **Topic overlap**: Some topics may be difficult to distinguish
- **Interpretation subjectivity**: Manual labeling introduces bias

### Planned Improvements
- **Dynamic Topic Models**: Capture topic evolution over time
- **Neural Topic Models**: Use embeddings for better semantic coherence
- **Hierarchical Models**: Multi-level topic structures
- **Interactive Tools**: Web interface for topic exploration

### Research Extensions
- **Cross-cultural comparisons**: Compare topics across different regions
- **Multi-platform analysis**: Extend to Twitter, Reddit, news comments
- **Longitudinal studies**: Track topic changes with robot deployment phases
- **Mixed-methods**: Combine with qualitative interviews for deeper insights

This topic modeling implementation provides a systematic approach to understanding public discourse about sidewalk delivery robots, complementing sentiment analysis with thematic insights.