# MP2 Reflection

## What worked

The pipeline loaded all five stories, created 86 chunks with a maximum length of 500 characters, and stored all 86 vectors in the `mp2_sherlock` collection. Paragraph- and heading-aware chunking keeps section information with story passages, while the 80-character overlap carries context across split passages. Validation cited the expected source for both predefined questions (2/2) and all three learner questions (3/3).

## What didn't work

Expected-fact matches varied across the questions: Q1 matched 1/5 phrases, Q2 matched 4/6, Q3 matched 3/3, Q4 matched 1/3, and Q5 matched 1/4. The validator checks literal phrases, so a phrase not counted may have been omitted or expressed differently. The lower counts for Q1, Q4, and Q5 need closer inspection even though each answer cited the expected story.

## What I'd change

I would review the retrieved excerpts and actual generated answers for Q1, Q4, and Q5. That would show whether the expected details were missing from retrieval or whether the answer phrasing differed from the validator's phrases. I would make a targeted retrieval or prompt adjustment only after that review, then rerun validation to compare source matches and fact coverage.

## One surprise

I was surprised by the gap between source matching and expected-fact matching: all five questions cited the expected story, but the validator matched 10 of 21 expected phrases overall. This showed me that retrieving the right source and expressing the expected details are separate outcomes to evaluate.
