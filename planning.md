# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

Student reviews and opinions on the Chemistry teaching faculty at Temple University. This is useful, as it allows distinctions between teachers' teaching style, grading, workload, and personality. 
This is unavailible through official channels, as professor feedback from students is not made public, and teaching styles as well as class workload is not shared anywhere on official university channels.
---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | https://www.ratemyprofessors.com/professor/1010415| Professor Feedback reviews | URL |
| 2 | https://www.ratemyprofessors.com/professor/2166653| Professor Feedback reviews | URL |
| 3 | https://www.ratemyprofessors.com/professor/1287533| Professor Feedback reviews | URL|
| 4 | https://www.ratemyprofessors.com/professor/733530| Professor Feedback reviews | URL |
| 5 | https://www.ratemyprofessors.com/professor/2423438| Professor Feedback reviews | URL |
| 6 | https://www.ratemyprofessors.com/professor/1041500| Professor Feedback reviews | URL |
| 7 | https://www.ratemyprofessors.com/professor/2315341| Professor Feedback reviews | URL |
| 8 | https://www.ratemyprofessors.com/professor/240055| Professor Feedback reviews | URL |
| 9 | https://www.ratemyprofessors.com/professor/2854648| Professor Feedback reviews | URL |
| 10 | https://www.ratemyprofessors.com/professor/2045650| Professor Feedback reviews | URL |
| 11 | https://www.ratemyprofessors.com/professor/1238947| Professor Feedback reviews | URL |
| 12 | https://www.ratemyprofessors.com/professor/1700141| Professor Feedback reviews | URL |


---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** 300-600 characters

**Overlap:** No overlap should be needed

**Reasoning:** Each student Review and/or comment is a pretty short sum of information that can be taken as a whole, meaning that the chunks collected will be 1 whole review, (~300-600 chars), and that their overlap does not have to be too great, as each chunk is a basically a distinct unconnected statement.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** all-mpnet-base-v2

**Top-k:** This project will need 2 top-k, a larger one, equal to 8 to collect information on each professor, and a second equal to 3 to return collected professor information

**Production tradeoff reflection:** I would weight the ability to understand reviews from different languages, be able to understand more sources of different lengths and types rather than all short reviews, and I would try to add more accuracy by collecting the sentiment of students comparatively with the grade they recieved in the class to see if such factors affected their opinions od the instructor.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Which professor teaches the concepts of Organic Chemsitry the best?| Steven Fleming|
| 2 | Which professor between (professor A) and (Professor B) has a better grading curve?| Professor B|
| 3 | What is the general opinion of (professor C)'s teaching style| Hard to understand and disorganized|
| 4 | Between these two professors (Prof D and E) which is better at explaining Physical Chemistry| Professor E|
| 5 | Which Professor is most leniant about late work| Professor B and E are both equally leniant|

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. There isn't the same amount of information about each professor, so some professors might show up statistically more or less based on the amount of availible informations, rather then viability/ fitting for the question.

2. While Rate my Professor's website isn't particularly noisy or messy, there is a lot of unnecessary information that will need to be parsed out

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->
     Stored as Week 1 Pipeline-1.png
---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:** I'll use Claude to generate a program that intakes a website URL, allows it to see the website, scan the text on the website and break it into the desired chunks made of exclusively the professor review text. Additionally, store the name of the professor in order to attach it to the metadata of the embedding chunks. Test that the Rate my Professor reviews are properly chunked by outputting them to make sure they match the actual reviews.

**Milestone 4 — Embedding and retrieval:** Use all-mpnet-base-v2 to embed the chunks into retrieveable vectors. In addition to the chunks, the Professor's name will need to be added to the metadata of the vector, as not all reviews will include the professor's name. This will be important so that the reviews of any professor will not be used for another.

**Milestone 5 — Generation and interface:** Use the stored information gathered from the website and feed it to an API model in order to generate a viable response to the provided prompt with the most relevent information collected from the websites.
