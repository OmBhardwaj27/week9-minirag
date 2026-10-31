# MP2 Reflection

## What worked

Paragraph-aware chunking keeps section headings attached to story passages, and the 80-character overlap carries nearby context across chunk boundaries. Offline checks loaded all five stories and produced 86 chunks, each no longer than 500 characters. Citations retain the source filename and section.

## What didn't work

All five questions retrieved and cited their expected story, but exact expected-fact matches were low for some answers: 1/5 for Q1, 1/3 for Q4, and 1/4 for Q5. The validator checks literal fact phrases, so these results point to answers that need closer inspection even when the source citation is correct.

## What I'd change

I would inspect the retrieved excerpts and generated responses for Q1, Q4, and Q5, then refine the answer instructions only if the expected facts are present in the context. I would rerun validation after any change.

## One surprise

I was surprised that all five questions cited the expected story, yet some answers still missed several expected facts. Q1 matched only 1 of 5 facts, and Q5 matched only 1 of 4. This showed me that retrieving the right story is not the same as giving a complete answer from it.
