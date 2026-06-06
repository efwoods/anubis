https://www.nltk.org/

Search for the following and create a list of:

Lexicon (the vocabulary)
catchphrases
preferred synonyms
achronisms or slang


Capture Syntax and Structure
punctuation habits
sentence length
Capitalization

Syntactic Features (rough list):
1. Syntactic Complexity MetricsThese track how sentences are put together and the cognitive effort required to read them.
   1. Mean Length of Sentence (MLS): The average number of words per sentence.Mean Length of T-Unit (MLT): A T-unit is a main clause plus any subordinate clauses attached to it. This measures the smallest independent grammatical units.
   2. Dependent Clauses per Clause (DC/C): The ratio of subordinate clauses to total clauses. Higher ratios indicate dense, complex writing (e.g., academic or legal).
   3. Coordinate Phrases per Clause (CP/C): Tracks how often a writer links equal ideas using conjunctions like "and" or "but."Parse Tree Depth: The maximum depth of a grammatical sentence diagram. 
   4. Deep trees indicate highly nested, complex grammar.
 
2. Lexical and Morphological DistributionsThese measure vocabulary variety, word patterns, and the grammatical parts of speech.
   1. Type-Token Ratio (TTR): The number of unique words (types) divided by the total number of words (tokens). High TTR means a rich vocabulary; low TTR means repetitive language.
   2. Part-of-Speech (POS) Density: The exact percentage of nouns, verbs, adjectives, adverbs, and pronouns. For example, action-oriented writing has a high verb density, while descriptive writing has high adjective density.
   3. Lexical Density: The ratio of content words (nouns, verbs, adjectives) to functional words (prepositions, articles, conjunctions).
   4. Noun-to-Verb Ratio: Heuristic indicating writing style. A high ratio indicates a formal, information-heavy style ("nominal style"). A low ratio indicates a conversational, action-driven style ("verbal style").

3. Structural and Structural Flow (Heuristics)These observe how text physically flows across paragraphs and formatting boundaries.
   1. Paragraph-to-Sentence Ratio: Tracks structural pacing (e.g., modern internet writing uses 1-2 sentences per paragraph; literature uses much more).
   2. Transition Density: The frequency of logical bridge words (e.g., however, therefore, conversely, furthermore).
   3. Punctuation Fingerprint: The specific distribution and frequency of commas, semicolons, dashes, ellipses, and exclamation marks per 1,000 words.
   4. Sentence Architecture Sequences: The sequential pattern of sentence structures (e.g., does the writer consistently follow a long, complex sentence with a short, punchy one?).
5. Information Theory and Randomness MetricsThese look at text purely as mathematical data to find hidden patterns.
   1. Lexical Entropy: Measures the predictability of word choices. 
   2. High entropy means the next word is highly unpredictable.
   3. N-gram Frequency Distribution: Tracking the recurrence of specific 2-word (bigram) or 3-word (trigram) sequences. 
   4. This catches unique personal phrasing habits (e.g., "at the end of the day").
   5. Burstiness: Measures whether specific rare words appear randomly throughout the text, or if they appear in tight, clustered bursts.
6. Readability Formulas (Composite Heuristics)These mathematical formulas combine sentence length and syllable counts to output a standardized grade level.
   1. Flesch-Kincaid Grade Level / Reading Ease: Measures words per sentence and syllables per word.
   2. Gunning Fog Index: Uses sentence length and the percentage of complex words (three or more syllables) to calculate reading difficulty.
   3. SMOG Index: Estimates the years of education needed to fully understand a piece of text based on polysyllabic word counts.

