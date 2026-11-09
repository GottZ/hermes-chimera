"""Text preparation before the template's G2P (the plugin's own step, switchable off).

The template's front end is faithful to its JS original and stumbles on text an LLM writes: "1,000,000" becomes
"one | zero | zero" (the comma ends a phrase), "Dr." becomes "drive" plus a phrase break, "café" loses its é,
decimals break at the point, and numbers from a million on are read digit by digit. ``prepare_text`` rewrites
such text into plain ASCII English the front end reads well. Steps, in this order (the order matters: "No." is
"number" only before a digit, so abbreviations go before numbers):

1. Unicode NFKC; typographic quotes and dashes to ASCII.
2. Abbreviations with a period to words, so the period no longer breaks the phrase. One that ends the text
   keeps a period (it also ended the sentence).
3. Numbers: thousands separators dropped; ordinals ("21st" -> "twenty first"), clock times ("3:30" -> "three
   thirty", "7:00" -> "seven o'clock", "10:05" -> "ten oh five"), a leading minus, decimals ("3.14" -> "three
   point one four") and integers below 10^15 to words (short scale, no "and"); larger integers stay digits
   (the front end reads them digit by digit).
4. Latin letters with diacritics to their base letters (ß -> ss, æ -> ae, ø -> o, œ -> oe, ł -> l, þ -> th,
   ð -> d). Other scripts are not transliterated: English is the only language the front end knows.
5. Any other character outside ``[A-Za-z0-9 '.,;:!?-]`` to a space; brackets to comma pauses; whitespace
   collapsed, no space before punctuation, no comma right before other punctuation or at either end.

Returns "" when no letter or digit remains (nothing speakable); never raises for a ``str``. The result is
ASCII and ``prepare_text(prepare_text(x)) == prepare_text(x)``.
"""

from __future__ import annotations

import re
import unicodedata

_TYPO = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "“": '"', "”": '"',
         "„": '"', "″": '"', "–": "-", "—": "-", "‒": "-", "―": "-", "−": "-",
         "‐": "-", "‑": "-"}

# Case-sensitive on purpose: "Mar." is March, "mar" is a word. Letter names are spelled the way the
# dictionary knows them ("ay em", "pee em").
_ABBREVIATIONS = {
    "Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miz", "Prof.": "Professor", "Jr.": "Junior",
    "Sr.": "Senior", "Mt.": "Mount", "vs.": "versus", "etc.": "et cetera", "e.g.": "for example", "i.e.": "that is",
    "a.m.": "ay em", "p.m.": "pee em", "A.M.": "ay em", "P.M.": "pee em", "U.S.": "U S", "U.K.": "U K",
    "Jan.": "January", "Feb.": "February", "Mar.": "March", "Apr.": "April", "Jun.": "June", "Jul.": "July",
    "Aug.": "August", "Sep.": "September", "Sept.": "September", "Oct.": "October", "Nov.": "November",
    "Dec.": "December", "approx.": "approximately", "Ave.": "Avenue", "Inc.": "Incorporated", "Ltd.": "Limited",
    "Co.": "Company", "Corp.": "Corporation", "Gen.": "General", "Gov.": "Governor", "Capt.": "Captain",
    "Lt.": "Lieutenant", "Sgt.": "Sergeant", "Col.": "Colonel", "Rev.": "Reverend", "Fig.": "Figure",
    "fig.": "figure", "Vol.": "Volume", "vol.": "volume", "Dept.": "Department",
}
_ABBR_RE = [(re.compile(r"(?<![A-Za-z.])" + re.escape(k)), v)
            for k, v in sorted(_ABBREVIATIONS.items(), key=lambda kv: -len(kv[0]))]
_MARK = ""      # private-use marker: "an abbreviation ended here" (never in the output)

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_SCALES = ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (10 ** 3, "thousand"))
_ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
            "twelve": "twelfth"}
_LIMIT = 10 ** 15

_LETTERS = {"ß": "ss", "æ": "ae", "Æ": "AE", "ø": "o", "Ø": "O", "œ": "oe", "Œ": "OE", "ł": "l", "Ł": "L",
            "þ": "th", "Þ": "TH", "ð": "d", "Ð": "D", "đ": "d", "Đ": "D", "ı": "i", "ħ": "h", "Ħ": "H"}


def cardinal(n: int) -> str:
    """Short-scale cardinal for 0 <= n < 10^15 ("one hundred five", "two thousand twenty six")."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + cardinal(n % 100))
    for value, name in _SCALES:
        if n >= value:
            return cardinal(n // value) + " " + name + ("" if n % value == 0 else " " + cardinal(n % value))
    raise ValueError(n)


def ordinal(n: int) -> str:
    """"twenty first", "one hundredth", "twelfth"."""
    words = cardinal(n).split(" ")
    last = words[-1]
    if last in _ORDINAL:
        last = _ORDINAL[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return " ".join(words[:-1] + [last])


def _sub_words(pattern: re.Pattern, fn, s: str) -> str:
    """Replace matches by words, with one space towards an adjacent letter ("MP3" -> "MP three"); ``fn``
    returns None to leave a match as it is."""
    out, pos = [], 0
    for m in pattern.finditer(s):
        out.append(s[pos:m.start()])
        words = fn(m)
        if words is None:
            out.append(m.group(0))
        else:
            left = " " if m.start() > 0 and s[m.start() - 1].isalpha() else ""
            right = " " if m.end() < len(s) and s[m.end()].isalpha() else ""
            out.append(left + words + right)
        pos = m.end()
    out.append(s[pos:])
    return "".join(out)


_NUMBER_SIGN = re.compile(r"(?<![A-Za-z.])[Nn]o\.(?=\s*\d)")
_SAINT = re.compile(r"(?<![A-Za-z.])St\.(?=\s+[A-Z])")
_STREET = re.compile(r"(?<![A-Za-z.])St\.")
_END_MARK = re.compile(_MARK + r"(?=\s*$)")


def _abbreviations(s: str) -> str:
    s = _NUMBER_SIGN.sub("number ", s)
    s = _SAINT.sub("Saint" + _MARK, s)
    s = _STREET.sub("Street" + _MARK, s)
    for pattern, words in _ABBR_RE:
        s = pattern.sub(words + _MARK, s)
    # the abbreviation's period survives only as the end of the whole text
    return _END_MARK.sub(".", s).replace(_MARK, "")


_THOUSANDS = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_ORDINAL_RE = re.compile(r"(?<!\d)(\d+)(?:st|nd|rd|th)\b")
_CLOCK = re.compile(r"(?<![\d:])(\d{1,2}):([0-5]\d)(?![\d:])")
_MINUS = re.compile(r"(?<![A-Za-z0-9.])-(?=\d)")
_DECIMAL = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?![\d.]*\d)")
_INTEGER = re.compile(r"\d+")


def _clock(m: re.Match) -> str:
    h, mm = int(m.group(1)), int(m.group(2))
    if mm == 0:
        return cardinal(h) + " o'clock"
    return cardinal(h) + (" oh " + _ONES[mm] if mm < 10 else " " + cardinal(mm))


def _small(digits: str) -> int | None:
    """The value if below 10^15, else None (checked on the string: Python refuses int() of very long ones)."""
    d = digits.lstrip("0") or "0"
    return int(d) if len(d) <= 15 else None


def _int_words(digits: str) -> str | None:
    n = _small(digits)
    return None if n is None else cardinal(n)


def _ordinal_words(m: re.Match) -> str | None:
    n = _small(m.group(1))
    return None if n is None else ordinal(n)


def _decimal_words(m: re.Match) -> str:
    whole = _int_words(m.group(1))
    return (m.group(1) if whole is None else whole) + " point " + " ".join(_ONES[int(c)] for c in m.group(2))


def _numbers(s: str) -> str:
    s = _THOUSANDS.sub(lambda m: m.group(0).replace(",", ""), s)
    s = _sub_words(_ORDINAL_RE, _ordinal_words, s)
    s = _sub_words(_CLOCK, _clock, s)
    s = _MINUS.sub("minus ", s)
    s = _sub_words(_DECIMAL, _decimal_words, s)
    return _sub_words(_INTEGER, lambda m: _int_words(m.group(0)), s)


def _base_letters(s: str) -> str:
    out = []
    for ch in s:
        if ch in _LETTERS:
            out.append(_LETTERS[ch])
            continue
        out.extend(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
    return "".join(out)


_OPEN = re.compile(r"[(\[{]")
_CLOSE = re.compile(r"[)\]}]")
_OTHER = re.compile(r"[^A-Za-z0-9 '.,;:!?-]")
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r" +([.,;:!?])")
_COMMA_BEFORE_PUNCT = re.compile(r",(?= ?[.,;:!?]| ?$)")
_LEADING = re.compile(r"^[ ,]+")
_ALNUM = re.compile(r"[A-Za-z0-9]")


def _clean(s: str) -> str:
    s = _OPEN.sub(", ", s)
    s = _CLOSE.sub(", ", s)
    s = _OTHER.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    prev = None
    while prev != s:
        prev = s
        s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
        s = _COMMA_BEFORE_PUNCT.sub("", s)
        s = _LEADING.sub("", s)
        s = _SPACES.sub(" ", s).strip()
    return s if _ALNUM.search(s) else ""


def prepare_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    s = "".join(_TYPO.get(c, c) for c in s)
    return _clean(_base_letters(_numbers(_abbreviations(s))))
