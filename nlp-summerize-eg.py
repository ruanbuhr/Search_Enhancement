from query_summarizer import *

# Create an instance of the QuerySummarize class
summarizer = QuerySummarize()

# Example text to summarize
text = "What are the causes of climate change, and how have human activities contributed to it since the 1800s?"

# Get keywords
keywords = summarizer.summerize(text)

# Output the keywords
print("Keywords:", keywords)