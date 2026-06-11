# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

Student reviews and opinions on the Chemistry teaching faculty at Temple University. This is useful, as it allows distinctions between teachers' teaching style, grading, workload, and personality. 
This is unavailible through official channels, as professor feedback from students is not made public, and teaching styles as well as class workload is not shared anywhere on official university channels.

---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
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

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:** Chunk size is about 300-600 characters, as this is the length of rate my professor reviews.

**Overlap:** I used 0 overlap, as each chunk created is its own standalone review, meaning that there is no need for overlap, as each chunk is entirely self contained.

**Why these choices fit your documents:** As explained above these choices fit as they best break down rate my professor, as it is review based, meaning that each review is short enough that it can be treated as its own chunk, and hence needs no overlap.

**Final chunk count:** 986

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:** all-mpnet-base-v2

**Production tradeoff reflection:** I would weight the ability to understand reviews from different languages, be able to understand more sources of different lengths and types rather than all short reviews, and I would try to add more accuracy by collecting the sentiment of students comparatively with the grade they recieved in the class to see if such factors affected their opinions od the instructor.

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:** I grounded the llm by giving it a set of strict rule in the overall prompt. The rules include various ways that the system can begin halucinating. It specifically specifies to only use information from the provided references, not to use outside information, not to make up or use unrelated or unprovided information.

**How source attribution is surfaced in the response:** The source attribution is shown via in text references to a list of sources it used. In this specific case, it shows that it got the information from RMP, and shows which professor the review was for, and which review it was on RMP's website, so that if needed, someone can find the exact review it was referencing.

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Which professor teaches the concepts of Organic Chemsitry the best?| Steven Fleming| Steven Fleming, followed by Gilbert| Partially Relevent| Accurate|
| 2 | Which professor between Steven Fleming and Serge Jasmin has a better grading curve?| Serge Jasmin| Possibly Jasmin, but not enough information to give a definitive answer| Relevent| Accurate|
| 3 | What is the general opinion of Fleming's teaching style| Clear, Content Heavy, Ocassionally dismissive.| Thoughrough, content heavy, not for everyone| Relevent| Accurate|
| 4 | Between these two professors: Dr. Kaur or Dr. Fleming which is better at explaining Organic Chemistry| Dr. kaur| Depends on their context, but Kaur is clearer| Relevent| Accurate|
| 5 | Which Professor is most leniant about late work| Professor Kaur| Not Enough information| Off-taget | Partially Accurate|

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:** What is the most recent opinionson professor fleming?

**What the system returned:** Returned a viable answer based on the provided chunks, however provided chunks are irrelevent to the question.

**Root cause (tied to a specific pipeline stage):** The issue is in my chunking. This is because when chunking our reviews, our program only collects the information from the actual review text and the professor's name, and doesn't include anything else, including review date, meaning our embedding does not include anything but the actual review context, and to which professor that review corresponds. This means that the retriever has no time information to retrieve from, meaning it retrieves basically random reviews. 

**What you would change to fix it:** For this case, I can go about fixing it in two ways. In the more simple case, I can inform the system that the higher the review number is, the further back the review was made. This works as RMP displays more recent reviews first and older ones later, meaning that this simple solution can workk well for this specific context issue. However, if I want a more complex solution that is more robust, I could increase the chunker's scope to also inclue the review's provided date, and other less relevent(in my opinion) information such as students final grades, numberical reviews, and tags. While this would be more complex to implement, if we then attach the additional information to each text review's metadata, our program can make more robust answers, while also solving the timeline issue.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:** One unique way in which the specs helped me during implemintation was learning the overall path beforehand. Before doing the planning, I wasn't completely sure on what the overall path connected to one another. Additionally, some steps were entirely foreign to me. Completing the planning.md spec sheet allowed me to learn the connections, and understand the steps fully before implementation, allowing me to more easily transfer between steps, as I understoon how to tie each step into the other.

**One way your implementation diverged from the spec, and why:** One way my implementation diverged from the spec was via additional information being added to the metadata, specifically the review's number on RMP's website. While this was not necesarilly needed, this change was made in order to make in order to simplify and make the LLM's output less confusing, more trackable, and more possible for fact checking the information that the answer is using to provide answes.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- *What I gave the AI:* I gave Claude my original plan on how to chunk the reviews, and asked it to create the system in which to both import the website's HTML, and to clean it up to the correct chunking.
- *What it produced:* It returned a disfuntional program that although could produce the desired results, would neither succeed most of the time, didn't store the chunks anywhere, and only allowed one source to be entered at a time. 
- *What I changed or overrode:* After figuring out why the program was not functioning, I changed the timing to ensure it always suceeded in loading the HTML, Stores the chunks, and added implemintation for passing in a .txt file reader that can run the chunk maker on all the provided RMP links at once, combining them all into one stored chunks file.

**Instance 2**

- *What I gave the AI:* I fed Claude the three project files that I had been working on, and app.py from the project guideline. I asked it to join together my three seperate Project files that contained the functionality of my program into one program that I could use in app.py to create the UI.
- *What it produced:* It produced a program that didn't just combine all the functionality file, but also directly added the app.py to the program. 
- *What I changed or overrode:* I changed this overall program into a seperate program that just ran the functionality without directly leading to the UI. I made this change for the main reason that I wanted to ensure that the code itself still worked without the UI, so  that I knew that I could more easily edit any issues with the code, rather then dealing with them being directly tied to the ui.
