import re

ENDOFTEXT = "<|endoftext|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
SPECIAL_TOKENS = (ENDOFTEXT, USER, ASSISTANT)


class SpecialTokensMixin:
    special = None

    def _base_vocab_size(self):
        raise NotImplementedError

    def _encode_ordinary(self, text):
        raise NotImplementedError

    def _decode_ordinary(self, ids):
        raise NotImplementedError

    def _specials(self):
        if self.special is None:
            self.special = {}
        return self.special

    def add_special_tokens(self, names=SPECIAL_TOKENS):
        specials = self._specials()
        for name in names:
            if not name or name in specials:
                continue
            specials[name] = self._base_vocab_size() + len(specials)
        self._special_inverse = None
        self._special_pattern = None
        return self

    def special_id(self, name):
        return self._specials().get(name)

    @property
    def special_ids(self):
        return frozenset(self._specials().values())

    @property
    def vocab_size(self):
        return self._base_vocab_size() + len(self._specials())

    def _inverse(self):
        if getattr(self, "_special_inverse", None) is None:
            self._special_inverse = {v: k for k, v in self._specials().items()}
        return self._special_inverse

    def encode(self, text, allowed_special=()):
        specials = self._specials()
        if allowed_special == "all":
            allowed = set(specials)
        else:
            allowed = set(allowed_special)
            unknown = allowed - set(specials)
            if unknown:
                raise ValueError(f"tokenizer has no special token(s): {sorted(unknown)}")
        if not allowed:
            return self._encode_ordinary(text)
        pattern = re.compile("(" + "|".join(re.escape(s) for s in sorted(allowed, key=len, reverse=True)) + ")")
        out = []
        for piece in pattern.split(text):
            if not piece:
                continue
            if piece in allowed:
                out.append(specials[piece])
            else:
                out.extend(self._encode_ordinary(piece))
        return out

    def decode(self, ids, skip_special=False):
        inverse = self._inverse()
        if not inverse:
            return self._decode_ordinary(ids)
        parts, run = [], []
        for i in ids:
            i = int(i)
            if i in inverse:
                if run:
                    parts.append(self._decode_ordinary(run))
                    run = []
                if not skip_special:
                    parts.append(inverse[i])
            else:
                run.append(i)
        if run:
            parts.append(self._decode_ordinary(run))
        return "".join(parts)

    def _with_special(self, d):
        if self._specials():
            d["special"] = dict(self._specials())
        return d

    def _restore_special(self, d):
        special = d.get("special") or {}
        expected = self._base_vocab_size()
        for offset, (name, sid) in enumerate(sorted(special.items(), key=lambda kv: kv[1])):
            if int(sid) != expected + offset:
                raise ValueError("special token ids must follow the base vocabulary contiguously")
        self.special = {k: int(v) for k, v in special.items()}
        self._special_inverse = None
        return self
