from typing import List, Tuple, Dict, Any

class BBoxNormalizer:
    """
    Handles coordinate transformations between image pixel coordinates
    and Layout Transformer normalized space [0, 1000].
    """

    @staticmethod
    def normalize_box(box: List[int], width: int, height: int) -> List[int]:
        """
        Normalizes a pixel bounding box [x0, y0, x1, y1] to the [0, 1000] integer scale.
        """
        if width <= 0 or height <= 0:
            return [0, 0, 0, 0]

        x0, y0, x1, y1 = box
        norm_x0 = int(1000 * (x0 / width))
        norm_y0 = int(1000 * (y0 / height))
        norm_x1 = int(1000 * (x1 / width))
        norm_y1 = int(1000 * (y1 / height))

        # Clamp between 0 and 1000 to prevent out-of-bounds positional embedding index
        return [
            max(0, min(1000, norm_x0)),
            max(0, min(1000, norm_y0)),
            max(0, min(1000, norm_x1)),
            max(0, min(1000, norm_y1))
        ]

    @staticmethod
    def denormalize_box(norm_box: List[int], width: int, height: int) -> List[int]:
        """
        Converts a [0, 1000] normalized bounding box back to pixel coordinates.
        """
        nx0, ny0, nx1, ny1 = norm_box
        x0 = int((nx0 / 1000.0) * width)
        y0 = int((ny0 / 1000.0) * height)
        x1 = int((nx1 / 1000.0) * width)
        y1 = int((ny1 / 1000.0) * height)
        return [max(0, x0), max(0, y0), min(width, x1), min(height, y1)]

    @staticmethod
    def align_subwords_with_boxes(
        words: List[str],
        boxes: List[List[int]],
        tokenizer,
        max_seq_length: int = 512
    ) -> Dict[str, Any]:
        """
        Aligns subword tokens with word-level bounding boxes.
        
        Why this is critical for LiLT / Layout Transformers:
        - OCR outputs word-level boxes (e.g., 'INV-2024-99' -> 1 bounding box).
        - Byte-Pair Encoding (BPE) breaks words into subwords:
          ['INV', '-', '2024', '-', '99'].
        - Every subword MUST inherit its parent word's 2D bounding box!
        - [CLS] (or <s>) and [SEP] (or </s>) special tokens receive [0, 0, 0, 0].
        
        Returns:
            Dictionary containing tokens, token_boxes, word_to_token_map, and token_to_word_map.
        """
        token_list: List[str] = []
        token_boxes: List[List[int]] = []
        word_to_token_indices: List[List[int]] = []
        token_to_word_idx: List[int] = []

        # Start token (<s> for RoBERTa / [CLS] for BERT)
        cls_token = getattr(tokenizer, "cls_token", "<s>") or "<s>"
        token_list.append(cls_token)
        token_boxes.append([0, 0, 0, 0])
        token_to_word_idx.append(-1)

        for w_idx, (word, box) in enumerate(zip(words, boxes)):
            # Tokenize word
            sub_tokens = tokenizer.tokenize(word)
            if not sub_tokens:
                sub_tokens = [tokenizer.unk_token or "<unk>"]

            token_indices_for_this_word = []
            for sub in sub_tokens:
                # Stop if reaching max length minus special end token
                if len(token_list) >= max_seq_length - 1:
                    break
                current_idx = len(token_list)
                token_list.append(sub)
                token_boxes.append(box)
                token_to_word_idx.append(w_idx)
                token_indices_for_this_word.append(current_idx)

            word_to_token_indices.append(token_indices_for_this_word)

            if len(token_list) >= max_seq_length - 1:
                break

        # End token (</s> for RoBERTa / [SEP] for BERT)
        sep_token = getattr(tokenizer, "sep_token", "</s>") or "</s>"
        token_list.append(sep_token)
        token_boxes.append([0, 0, 0, 0])
        token_to_word_idx.append(-1)

        return {
            "tokens": token_list,
            "token_boxes": token_boxes,
            "word_to_token_indices": word_to_token_indices,
            "token_to_word_idx": token_to_word_idx
        }
