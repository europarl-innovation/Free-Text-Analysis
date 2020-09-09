#' Demo script to manipulate the free text input file: sentimental analysis, wordclouds 
#' and topic clustering
#' 
#'  
#' Author: Charalampos Moschopoulos
#' Date: May 2020
#' 
#' Description:
#'  
#' 
#' NOTE:
#' - removedAcronymss.txt contains acronyms to be removed.
#' - removedCommonWords contains usual words that should be removed  
#'
#' 
# ###################################################################################
#' Preparation: setting working path,libraries needed
# ###################################################################################

  
setwd("~/EP_free-text analysis/")
library("RColorBrewer")
library("NLP")
library("tm")
library("topicmodels")
library("LDAvis")
library("lda")
library("quanteda")
library("SnowballC")
library("wordcloud")
library("sentimentr")
library("ggplot2")
library("qdap")
library("RWeka")
library("servr")

file_name="total_text.csv"
# ###################################################################################
#' Create a customized stopwords list
# ###################################################################################
#CommonWords <- read.table("removedCommonWordsITN.txt",header = FALSE, stringsAsFactors = FALSE,sep="\n")
#Acronyms <- read.table("removedAcronyms.txt",header = FALSE, stringsAsFactors = FALSE,sep="\n")
#Acronyms <- tolower(Acronyms$V1)

stop_words <- c(stopwords(source ="smart"),stopwords("english"))

# ###################################################################################
#' Read the file and calculate the sentiment
# ###################################################################################
text<-read.csv(file_name, header=FALSE, stringsAsFactors = FALSE, strip.white=TRUE, sep=";")
sentiment <- sentiment_by(text$V1)

summary(sentiment$ave_sentiment)

qplot(sentiment$ave_sentiment,binwidth=0.1,   geom="histogram",main="Review Sentiment Histogram", xlab = "Sentiment score")

#text <-iconv(enc2utf8(text),sub="byte")

corpus <- VCorpus(VectorSource(as.character(text)))

# ###################################################################################
#' Preprocessing of text
# ###################################################################################
docs <- tm_map(corpus, removeNumbers)  
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeWords, stop_words)
docs <- tm_map(docs, removePunctuation) 
docs <- tm_map(docs, stripWhitespace) 

#inspect(docs[10])
#docs <- tm_map(docs, stemDocument) 


docs <- tm_map(docs, PlainTextDocument) 

# ###################################################################################
#' Finding the most popular terms and create a wordcloud
# ###################################################################################

dtm <- DocumentTermMatrix(docs) 

freq <- sort(colSums(as.matrix(dtm)), decreasing=TRUE) 
wf <- data.frame(word=names(freq), freq=freq,row.names = NULL)   
#write.csv(wf,"popular_terms.csv",row.names = FALSE)

set.seed(1234)
dev.off()
dev.new(width = 1000, height = 1000, unit = "px")
wordcloud(words = wf$word, freq = wf$freq, min.freq = 2,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

# ###################################################################################
#' Prepare data for the LDA
# ###################################################################################

# tokenize on space and output as a list:
doc.list <- strsplit(as.character(docs), "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

# remove terms that are stop words or occur fewer than 100 times:
del <- names(term.table) %in% stop_words | term.table < 10 
term.table <- term.table[!del]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

# Compute some statistics related to the data set:
D <- length(documents)  # number of documents 
W <- length(vocab)  # number of terms in the vocab 
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document 
N <- sum(doc.length)  # total number of tokens in the data 
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus 

# ###################################################################################
#' Apply LDA algorithm
# ###################################################################################
# MCMC and model tuning parameters:
K <- 10
G <- 5000
alpha <- 0.02
eta <- 0.02

# Fit the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2 - t1 

# ###################################################################################
#' Visualise the  LDA results with LDAvis
# ###################################################################################
#estimates of the document-topic distributions (theta) and the set of topic-term distributions (phi) 
theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

Proposal_Abstracts <- list(phi = phi,
                           theta = theta,
                           doc.length = doc.length,
                           vocab = vocab,
                           term.frequency = term.frequency)

# create the JSON object to feed the visualization:
json <- createJSON(phi = Proposal_Abstracts$phi, 
                   theta = Proposal_Abstracts$theta, 
                   doc.length = Proposal_Abstracts$doc.length, 
                   vocab = Proposal_Abstracts$vocab,
                   term.frequency = Proposal_Abstracts$term.frequency)

serVis(json, out.dir = 'visTotaltext', open.browser = FALSE)
