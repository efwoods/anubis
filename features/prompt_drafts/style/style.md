 @style.ipynb @src/anubis/utils/dataset/authenticity_evaluator.py I am attempting to identify the list of quantifiable metrics that define each generated text and know that there is a statistically significant
  difference between the baseline and the generated responses with respect to those features and no statistically significant difference between the generated text and the ground truth texts (tweets in this instance)
  with respect to the list of features. I don't expect all the features to be useful in classifying whether a new text is alike to the ground truth and dislike to the baseline regular chatgpt answers. I expect there
  to be multicollinearity within the features given that I have multiple features that measure the same category. I should identify those features and choose a single feature to be representative.

I need to perform the same text processing and feature calculation for the generated responses as the @style.ipynb notebook. The @style.ipynb notebook is currently incorrect.  I am using the notebook to verify the applicability of the process of feature enumeration and comparison to be certain the process will be appropriate at scale for all users.

I want to compute the metrics for baseline once (unchanged) then store and retrieve the representative values for comparison when responding. I want to compute and update the ground_truth representative values whenever there is new ground-truth data present. 

I am using Mahalanobis Distance as a metric.
I need to verify the distribution of the data so as to be able to use the Mahalanobis distance. 

If the distribution is not chi squared, I understand I may not use the Mahalanobis Distance effectively.
I do not want to use spaCy in addition to nltk.
I do not want to use Vader sentiment scores. 

I do not want to use z scores. I want to normalize the scores between zero and one. 

# Impute NaNs with per-column median, then z-score using global stats so every
# feature contributes on a comparable scale.
X = features_df[FEATURE_COLS].copy()
X = X.fillna(X.median(numeric_only=True))
mu_g, sd_g = X.mean(), X.std(ddof=0).replace(0, 1.0)
Z = (X - mu_g) / sd_g
Z["corpus"] = features_df["corpus"].values


I need to return the numerical feature values, the variables of those values should be self-commenting rather than acronyms, and I need to determine not only the threshold at which the computed mahalanobis distance value becomes an outlier, but the actual value itself (so I know if the generated text is an outlier to the baseline and an outlier to the ground-truth). I want to use the 33 features as listed and (optionally) the first 20 principal components of the embeddings of the generated text. 


# Future Context ( not action items ):
The end goal is to assert the following statements about the generated text:
There is no statistically significant difference between the generated text and the ground truth responses N% of the time (the majority) with respect to the first 20 principal components and the listed features.
There is a statistically significant difference between the generated text and the baseline (ChatGPT) responses N% of the time (the majority) with respect to the first 20 principal components and the listed features. 

I will then need to create a dataset of facts from database for the avatar and test the responses for factual correctness.

I will need to identify and test for emotional triggers (I am already assigning sentiment analysis to the responses [measuring the emotional content])

I will need to identify and test for known relationships

I will need to identify and test the behavioral choices during conversations (using a predetermined script of events (a movie script or video game dialogue)) and apply this process to all avatars

This will allow for the measurement (awareness) of likeness of written syntax, accuracy of factual knowledge, accuracy of relationships, likeness of behavioral choices.

I will need to train and attach an adapter to allow for the capture of behavioral choices and grammar of the avatar. 

(Identifying includes collecting personal accounts, local, or social media data and using a model with structured output to search for the emotional triggers, the social relationships, and situaltional choices for few-shot prompting)

I will then be able to edit the system prompt, improve retrieval augmented generation, train and attach adapters and measure the before-and-after effect on the similarity of the generated responses to the generated text and the dissimilarity of the generated responses to the baseline chatgpt text then modify the amount of data that is collected, the preprocessing of that data, the means of retrieval augmented generation, the system prompt of instructions, the LLM that is used for inference, and the adapter (and hyper parameters) that is attached to the model until the factual and relational false positives and false negatives are minimized, the emotional sentiment is similar to the ground truth data in alignment with the previous measurements, there is no significant difference between the grammatical structure and the ground-truth statements, and the choices made are dissimilar from the choices made from the real individual based upon known works.

The avatar talks and behaves like the real person. The avatar knows what the real person knows. The avatar has relationships like the real individual. The avatar has the same emotional feeling as the real individual.

The avatar sounds and looks like the individual. The avatar moves like the individual. The avatar thinks like the individual.

# Application:
Man in the middle responses (email, text, tweet) (This person told you this; What do you want to say to her? (test desired response against what would have been generated; when there is no significant difference, or the user accepts the response, then the response is auto-handled as if a secretary handling the public))
Conversation suggestions (live transcriptions; stream of conversation fed to avatar; ignore, respond, notify; generate suggestions based upon the context of the conversation to date)

# Reasons
nostalgic purposes
knowing your heritage
wanting to be remembered

# Mediums
Data analytics 
Thought-to-text/text-to-thought
Thought-to-motion: important

Continue to message and tweet to anyone based upon shared memories from a verified primary source, local media, and social media