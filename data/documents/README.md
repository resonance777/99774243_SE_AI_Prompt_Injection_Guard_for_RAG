# Documents

Synthetic documents used as the retrieval corpus.

`invoice_INV-2026-1193.txt` is a fabricated supplier invoice. Both the company
and the transaction are invented; no real invoice was used, and none could be.

It carries one deliberate defect: position 3 contains a passage addressed not
to the reader but to the reviewing system, instructing it to suppress its
findings and approve the document. It is the attack this project defends
against, and the reason the corpus needs a document that contains one.
