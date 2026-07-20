#
# Tokenize the current sentence.
#
sentence_token_ids = Tokenizer.tokenize(sentence)

#
# -------------------------------------------------------------------------
# Handle oversized sentence.
#
# If a single sentence exceeds the configured chunk size,
# preserving the sentence boundary is impossible.
#
# In this case:
#   1. Flush the current chunk (if any).
#   2. Split only this sentence into fixed-size token chunks.
# -------------------------------------------------------------------------
#
if len(sentence_token_ids) > self.chunk_size:

    #
    # Flush the current semantic chunk before processing
    # the oversized sentence.
    #
    if current_tokens:

        chunks.append(
            self._create_chunk(
                chunk_id=chunk_id,
                document=document,
                token_ids=current_tokens,
                start_token=start_token,
            )
        )

        chunk_id += 1

        start_token += len(current_tokens)

        current_tokens.clear()

    #
    # Split the oversized sentence using fixed-size token chunks.
    #
    for start in range(
        0,
        len(sentence_token_ids),
        self.chunk_size,
    ):

        end = min(
            start + self.chunk_size,
            len(sentence_token_ids),
        )

        token_slice = sentence_token_ids[start:end]

        chunk_text = Tokenizer.detokenize(
            token_slice
        )

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                document_name=document.file_name,

                start_token=start_token,
                end_token=start_token + len(token_slice) - 1,

                text=chunk_text,

                token_count=len(token_slice),

                strategy="Semantic",
            )
        )

        chunk_id += 1

        start_token += len(token_slice)