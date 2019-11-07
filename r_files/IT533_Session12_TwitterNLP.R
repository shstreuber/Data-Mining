## Searching through text isn't all that hard.  
## We will implement an old-school vector space model of information retrieval, which
## is kind of how Hadoop still works today.
## Code thanks to https://www.r-bloggers.com/build-a-search-engine-in-20-minutes-or-less/

## We need these packages

install.packages("tm")
install.packages("SnowballC")

## We assume that doc1-doc7 below is our available document pool.

doc1 <- "Stray cats are running all over the place. I see 10 a day!"
doc2 <- "Cats are killers. They kill billions of animals a year."
doc3 <- "The best food in Columbus, OH is the North Market."
doc4 <- "Brand A is the best tasting cat food around. Your cat will love it."
doc5 <- "Buy Brand C cat food for your cat. Brand C makes healthy and happy cats."
doc6 <- "The Arnold Classic came to town this weekend. It reminds us to be healthy."
doc7 <- "I have nothing to say. In summary, I have told you nothing."

## This is the big file system, also called a "corpus."

doc.list <- list(doc1, doc2, doc3, doc4, doc5, doc6, doc7)
N.docs <- length(doc.list)
names(doc.list) <- paste0("doc", c(1:N.docs))

## And this is our search query

query <- "Healthy cat food"

## The Corpus class is a fundamental data structure in tm.
## We treated the query like any other document. It is, after all, just another string of text. 
## Queries are not typically known a priori, but in the processing steps that follow, 
## we will pretend like we knew ours in advance to avoid repeating steps.

library(tm)
library(SnowballC)
my.docs <- VectorSource(c(doc.list, query))
my.docs$Names <- c(names(doc.list), "query")
print(my.docs)

my.corpus <- Corpus(my.docs)
my.corpus
inspect(my.corpus)

###################### Transforming Text #############################

## One of the nice things about the Corpus class is the tm_map function, which cleans and 
## standardizes documents within a Corpus object. Below are some of the transformations.

getTransformations()

## First, let's get rid of punctuation.

my.corpus <- tm_map(my.corpus, removePunctuation)
my.corpus
inspect(my.corpus[1])

## Stray cats are running all over the place I see 10 a day

## Suppose we don't want to count "cats" and "cat" as two separate words. Then we will 
## use the stemDocument transformation to implement the famous Porter Stemmer algorithm. 
## To use this particular transformation, first load the Snowball package.

library(SnowballC)
my.corpus <- tm_map(my.corpus, stemDocument)
my.corpus
inspect(my.corpus[1])

## Stray cat are run all over the place I see 10 a day

## Finally, remove numbers and any extra white space.

my.corpus <- tm_map(my.corpus, removeNumbers)
my.corpus <- tm_map(my.corpus, tolower)
my.corpus <- tm_map(my.corpus, stripWhitespace)
inspect(my.corpus[1])

## stray cat are run all over the place i see a day

##################### Vector Space Model #############################

## Here's a trick that's been around for a while: represent each document as a vector in 
## \( \mathcal{R}^N \) (with \( N \) as the number of words) and use the angle \( \theta \) 
## between the vectors as a similarity measure. Rank by the similarity of each document to 
## the query and you have a search engine.

## One of the simplest things we can do is to count words within documents. This naturally 
## forms a two dimensional structure, the term document matrix, with rows corresponding to 
## the words and the columns corresponding to the documents. As with any matrix, we may think 
## of a term document matrix as a collection of column vectors existing in a space defined 
## by the rows. The query lives in this space as well, though in practice we wouldn't know 
## it beforehand.

term.doc.matrix.stm <- TermDocumentMatrix(my.corpus)
inspect(term.doc.matrix.stm[0:14, ])

## The matrices in tm are of type Simple Triplet Matrix where only the triples \( (i, j, value) \) 
## are stored for non-zero values. To work directly with these objects, you may use install the 
## slam [4] package. We bear some extra cost by making the matrix "dense" (i.e., storing all the zeros).

term.doc.matrix <- as.matrix(term.doc.matrix.stm)

cat("Dense matrix representation costs", object.size(term.doc.matrix), "bytes.\n", 
    "Simple triplet matrix representation costs", object.size(term.doc.matrix.stm), 
    "bytes.")

## In term.doc.matrix, the dimensions of the document space are simple term frequencies. 
## This is fine, but other heuristics are available. For instance, rather than a linear increase 
## in the term frequency \( tf \), perhaps \( \sqrt(tf) \) or \( \log(tf) \) would provide a more 
## reasonable diminishing returns on word counts within documents.
## Rare words can also get a boost. The word "healthy" appears in only one document, whereas 
## "cat" appears in four. A word's document frequency \( df \) is the number of documents that 
## contain it, and a natural choice is to weight words inversely proportional to their \( df \)s. 
## As with term frequency, we may use logarithms or other transformations to achieve the desired 
## effect.
## The tm() function weightTfIdf offers one variety of tfidf weighting, but below we build our own. 
## For both the document and query, we choose tfidf weights of \( (1 + \log_2(tf)) \times \log_2(N/df) \), 
## which are defined to be \( 0 \) if \( tf = 0 \). Note that whenever a term does not occur in 
## a specific document, or when it appears in every document, its weight is zero.
## We implement this weighting function across entire rows of the term document matrix, and 
## therefore our tfidf function must take a term frequency vector and a document frequency scalar as inputs.

get.tf.idf.weights <- function(tf.vec, df) {
  # Computes tfidf weights from a term frequency vector and a document
  # frequency scalar
  weight = rep(0, length(tf.vec))
  weight[tf.vec > 0] = (1 + log2(tf.vec[tf.vec > 0])) * log2(N.docs/df)
  weight
}

cat("A word appearing in 4 of 6 documents, occuring 1, 2, 3, and 6 times, respectively: \n", 
    get.tf.idf.weights(c(1, 2, 3, 0, 0, 6), 4))

## Using apply, we run the tfidf weighting function on every row of the term document matrix. 
## The document frequency is easily derived from each row by the counting the non-zero entries 
## (not including the query).

get.weights.per.term.vec <- function(tfidf.row) {
  term.df <- sum(tfidf.row[1:N.docs] > 0)
  tf.idf.vec <- get.tf.idf.weights(tfidf.row, term.df)
  return(tf.idf.vec)
}

tfidf.matrix <- t(apply(term.doc.matrix, c(1), FUN = get.weights.per.term.vec))
colnames(tfidf.matrix) <- colnames(term.doc.matrix)

tfidf.matrix[0:3, ]

################### Implementing Cosine Similarity #########################

## Remember Cosine Similarity?  It's baaaaaaaaaaaaaaaack!!!
## A benefit of being in the vector space \( \mathcal{R}^N \) is the use of its dot product. 
## For vectors \( a \) and \( b \), the geometric definition of the dot product is 
## \( a \cdot b = \vert\vert a\vert\vert \, \vert\vert b \vert \vert \cos \theta \), where 
## \( \vert\vert \cdot \vert \vert \) is the euclidean norm (the root sum of squares) and 
## \( \theta \) is the angle between \( a \) and \( b \).
## In fact, we can work directly with the cosine of \( \theta \). For \( \theta \) in the interval 
## \( [-\pi, -\pi] \), the endpoints are orthogonality (totally unrelated documents) and the 
## center, zero, is complete collinearity (maximally similar documents). We can see that the 
## cosine decreases from its maximum value of \( 1.0 \) as the angle departs from zero in 
## either direction.

angle <- seq(-pi, pi, by = pi/16)
plot(cos(angle) ~ angle, type = "b", xlab = "angle in radians", main = "Cosine similarity by angle")

## We may furthermore normalize each column vector in our tfidf matrix so that its norm is one. 
## Now the dot product is \( \cos \theta \).

tfidf.matrix <- scale(tfidf.matrix, center = FALSE, scale = sqrt(colSums(tfidf.matrix^2)))
tfidf.matrix[0:3, ]

## Keeping the query alongside the other documents let us avoid repeating the same steps. 
## But now it's time to pretend it was never there.

query.vector <- tfidf.matrix[, (N.docs + 1)]
tfidf.matrix <- tfidf.matrix[, 1:N.docs]

## With the query vector and the set of document vectors in hand, it is time to go after the 
## cosine similarities. These are simple dot products as our vectors have been normalized to 
## unit length.
## Recall that matrix multiplication is really just a sequence of vector dot products. 
## The matrix operation below returns values of \( \cos \theta \) for each document vector 
## and the query vector.

doc.scores <- t(query.vector) %*% tfidf.matrix

## With scores in hand, rank the documents by their cosine similarities with the query vector.

results.df <- data.frame(doc = names(doc.list), score = t(doc.scores), text = unlist(doc.list))
results.df <- results.df[order(results.df$score, decreasing = TRUE), ]

## Let's try this out:

options(width = 2000)
print(results.df, row.names = FALSE, right = FALSE, digits = 2)

## Our "best" document, at least in an intuitive sense, comes out ahead with a score nearly 
## twice as high as its nearest competitor. Notice however that this next competitor has nothing 
## to do with cats. This is due to the relative rareness of the word "healthy" in the documents 
## and our choice to incorporate the inverse document frequency weighting for both documents 
## and query. Fortunately, the profoundly uninformative document 7 has been ranked dead last.

#############################################################################################
################# Text Retrieval with Twitter ##################################################
## NOTE:  You will need your own Twitter account and a completed OAuth Profile.  See:
## https://twittercommunity.com/t/oauth-authentication-with-twitter-api-via-r-failed/390

install.packages("twitteR")
install.packages("wordcloud")
install.packages("tm")
install.packages("qdap")
install.packages("SnowballC")
install.packages("RColorBrewer")
library(twitteR)
library(wordcloud)
library(tm)
library(qdap)
library(SnowballC)
library(RColorBrewer)

## The following information will have to from your Twitter API developer screen.
## You will be using the course information if you a student in my class.

consumer_key <- 'your_consumer_key'
consumer_secret <- 'your_consumer_secret'
access_token <- 'your_access_token'
access_secret <- 'your_access_secret'

setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)


################## Text Retrieval and Wordcloud ###############################
## Code (with some changes) from http://davetang.org/muse/2013/04/06/using-the-r_twitter-package/
###############################################################################

library(twitteR)

cats <- searchTwitter("#cats", n=100)
## should get 10
length(cats)
## [1] 10
str(cats [1:1])
## This is what the Twitter attributes look like.  Awesome, huh?

## Twitter Stream to DataFrame for other processing and because we can
cats_df <- twListToDF(cats)
cats_df

## Setting up the text for processing into the bag of words
cats_text <- sapply(cats, function(x) x$getText())
cats_text

## Create corpus; we use VectorSource because it interprets each element of the vector x as a document.
cats_text_corpus <- Corpus(VectorSource(cats_text))
inspect_text(cats_text_corpus)

## Clean up with the tm package
getTransformations()

## FIRST: Transform special characters to readable latin1. sapply is used to apply the function over the vector
cats_text_corpus$content <- sapply(cats_text_corpus$content,function(row) iconv(row, "latin1", "ASCII", sub=""))
inspect_text(cats_text_corpus)

## Make all lowercase
cats_text_corpus <- tm_map(cats_text_corpus, content_transformer(tolower))
inspect_text(cats_text_corpus)

## Remove punctuation
cats_text_corpus <- tm_map(cats_text_corpus, removePunctuation)
inspect_text(cats_text_corpus)

## Remove stopwords
cats_text_corpus <- tm_map(cats_text_corpus, function(x)removeWords(x,stopwords()))
inspect_text(cats_text_corpus)

## Strip extra white space
cats_text_corpus <- tm_map(cats_text_corpus, stripWhitespace)
inspect_text(cats_text_corpus)

## Tadah!  Build the wordcloud and ignore the errors!
wordcloud(cats_text_corpus)

## Use word stems only, so that "cats" and "cat" are counted as the same word
## What does this do to the wordcloud?
cats_text_corpus <- tm_map(cats_text_corpus, stemDocument)
inspect_text(cats_text_corpus)
wordcloud(cats_text_corpus)

## Lastly, we are using the RColorBrewer package to make the wordcloud colorful

library(RColorBrewer)
wordcloud(cats_text_corpus, colors=brewer.pal(8, "Dark2"))

################# A User's Term Frequency ########################################
## The code below is from https://www.r-bloggers.com/using-r-to-find-obamas-most-frequent-twitter-hashtags/
## With some fixes that made it possible to run in later versions of twitteR

## tw = userTimeline("BarackObama", cainfo = x1, n = 3200)
tw = userTimeline("BarackObama", n = 3200)
tw = twListToDF(tw)
vec1 = tw$text

extract.hashes = function(vec){
  
  ##  hash.pattern = "#[[:alpha:]]+"
  hash.pattern = "#[[:alnum:]]+"
  have.hash = grep(x = vec, pattern = hash.pattern)
  
  hash.matches = gregexpr(pattern = hash.pattern,
                          text = vec[have.hash])
  extracted.hash = regmatches(x = vec[have.hash], m = hash.matches)
  
  df = data.frame(table(tolower(unlist(extracted.hash))))
  colnames(df) = c("tag","freq")
  df = df[order(df$freq,decreasing = TRUE),]
  return(df)
}

dat = head(extract.hashes(vec1),50)
dat2 = transform(dat,tag = reorder(tag,freq))

library(ggplot2)

p = ggplot(dat2, aes(x = tag, y = freq)) + geom_bar(stat = "identity", fill = "blue")
p + coord_flip() + labs(title = "Hashtag frequencies in the tweets of the Obama team (@BarackObama)")

####################### Some other fun with Twitter #############################
## Code from https://www.r-bloggers.com/playing-with-twitter-data/
## With fixes where necessary

install.packages("lubridate")
install.packages("dplyr")
install.packages("cowplot")
library(lubridate)
library(dplyr)
library(cowplot)

tw = searchTwitter('#cats', n = 250, since = '2016-04-01')
d = twListToDF(tw)
str(d)

## Put in local time
d$created <- as.POSIXct(d$created, tz="GMT")
d$created = with_tz(d$created, 'America/Chicago')

timeDist = ggplot(d, aes(created)) + 
  geom_density(aes(fill = isRetweet), alpha = .5) +
  scale_fill_discrete(guide = 'none') +
  xlab('All tweets')

## Focus on one particular day

dayOf = filter(d, mday(created) == 22)
timeDistDayOf = ggplot(dayOf, aes(created)) + 
  geom_density(aes(fill = isRetweet), adjust = .25, alpha = .5) +
  theme(legend.justification = c(1, 1), legend.position = c(1, 1)) +
  xlab('Day-of tweets')
cowplot::plot_grid(timeDist, timeDistDayOf)

## Status Updates coming from
par(mar = c(3, 3, 3, 2))
d$statusSource = substr(d$statusSource, 
                        regexpr('>', d$statusSource) + 1, 
                        regexpr('</a>', d$statusSource) - 1)
dotchart(sort(table(d$statusSource)))
mtext('Number of tweets posted by platform')

## Emotional Valence
## Split into retweets and original tweets
sp = split(d, d$isRetweet)
orig = sp[['FALSE']]
## Extract the retweets and pull the original author's screenname
rt = mutate(sp[['TRUE']], sender = substr(text, 5, regexpr(':', text) - 1))

pol = 
  lapply(orig$text, function(txt) {
    # strip sentence enders so each tweet is analyzed as a sentence,
    # and +'s which muck up regex
    gsub('(\\.|!|\\?)\\s+|(\\++)', ' ', txt) %>%
      # strip URLs
      gsub(' http[^[:blank:]]+', '', .) %>%
      # calculate polarity
      polarity()
  })
orig$emotionalValence = sapply(pol, function(x) x$all$polarity)

## As reality check, what are the most and least positive tweets
orig$text[which.max(orig$emotionalValence)]
orig$text[which.min(orig$emotionalValence)]

## How does emotionalValence change over the day?
filter(orig, mday(created) == 22) %>%
  ggplot(aes(created, emotionalValence)) +
  geom_point() + 
  geom_smooth(span = .5)

## Do happier tweets get retweeted more?
ggplot(orig, aes(x = emotionalValence, y = retweetCount)) +
  geom_point(position = 'jitter') +
  geom_smooth()

## Emotional content
## I'm using qdap's polarity function straight out of the box to examine the emotional valence 
## of each tweet. One nice thing about that function is that it returns the positive and negative 
## words found in each text, which allows me to A) tabulate the most-used positively and negatively 
## valenced words, and B) to strip those words from positive and negative tweets to see what is 
## being talked about in positive and negative ways, without interference from the emotionally-charged words themselves.

polWordTables = 
  sapply(pol, function(p) {
    words = c(positiveWords = paste(p[[1]]$pos.words[[1]], collapse = ' '), 
              negativeWords = paste(p[[1]]$neg.words[[1]], collapse = ' '))
    gsub('-', '', words)  # Get rid of nothing found's "-"
  }) %>%
  apply(1, paste, collapse = ' ') %>% 
  stripWhitespace() %>% 
  strsplit(' ') %>%
  sapply(table)

par(mfrow = c(1, 2))
invisible(
  lapply(1:2, function(i) {
    dotchart(sort(polWordTables[[i]]), cex = .8)
    mtext(names(polWordTables)[i])
  }))

