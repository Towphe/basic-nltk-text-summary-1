import nltk
from nltk.tokenize import sent_tokenize
from util import create_freq_tab, score_sentences, find_avg_score, gen_summary

# Credits to
# https://becominghuman.ai/text-summarization-in-5-steps-using-nltk-65b21e352b65

# for much more complex and specialized tasks, generate your own models.
# NOTE: this only chain major sentences together, it doesn't create new sentences.
# look into abstracted text summary for that.

# open text file
f = open("text.txt", "r", encoding="utf8")

# extract file text content
text = f.read()

# close text file
f.close()

# create a frequency table of words in a text
freq_table = create_freq_tab(text)

# split text into sentences using nltk
sentences = sent_tokenize(text)

# score each sentence by its words; use quantity as `score`
sentence_scores = score_sentences(sentences, freq_table)

# calculate threshold depending on sentence scores
threshold = find_avg_score(sentence_scores)

# generate text summary
# adjust k accordingly; the higher k is, the shorter and nitpicky the result is
k = 1.2
summary = gen_summary(sentences, sentence_scores, k * threshold)

# create new output file
o = open("output.txt", "w")

# write summary text to output file
o.write(summary)

# close file.
o.close()