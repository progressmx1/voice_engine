"""Text normalisation for speech: turns written text into the words a person would say.

`normalize("Meet Dr. Lee at 3:30 pm on Mar 5, 2025 — bring $12.50 & a 2 kg bag.")` gives
"Meet Doctor Lee at three thirty P M on March fifth, twenty twenty-five, bring twelve dollars
and fifty cents and a two kilograms bag."

It handles markdown left in LLM replies, URLs and emails, abbreviations, currency, dates,
times, percentages, units, ordinals, years and decades, fractions, ranges, plain numbers and
symbols. Rules run in a fixed order (most specific first) so that, for example, a date is
read as a date before its digits are read as numbers. Numbers are read in American English
without "and" ("one hundred twenty-three"). Ambiguous slash dates default to month/day/year.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# --- Numbers --------------------------------------------------------------------------------

ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = ["", "thousand", "million", "billion", "trillion", "quadrillion"]
IRREGULAR_ORDINALS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
    "nine": "ninth", "twelve": "twelfth",
}


def _under_thousand(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    words = [f"{ONES[hundreds]} hundred"] if hundreds else []
    if rest >= 20:
        tens, ones = divmod(rest, 10)
        words.append(TENS[tens] + (f"-{ONES[ones]}" if ones else ""))
    elif rest or not hundreds:
        words.append(ONES[rest])
    return " ".join(words)


def cardinal(n: int) -> str:
    """123 -> "one hundred twenty-three"; negative numbers start with "minus"."""
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 1000:
        return _under_thousand(n)
    groups = []
    scale = 0
    while n:
        n, group = divmod(n, 1000)
        if group:
            name = SCALES[scale] if scale < len(SCALES) else ""
            groups.append(_under_thousand(group) + (f" {name}" if name else ""))
        scale += 1
    return " ".join(reversed(groups))


def ordinal(n: int) -> str:
    """21 -> "twenty-first", 100 -> "one hundredth"."""
    words = cardinal(n)
    head, sep, last = words.rpartition(" ")
    stem, dash, unit = last.rpartition("-")
    if unit in IRREGULAR_ORDINALS:
        unit = IRREGULAR_ORDINALS[unit]
    elif unit.endswith("y"):
        unit = unit[:-1] + "ieth"
    else:
        unit += "th"
    return head + sep + stem + dash + unit


def year(n: int) -> str:
    """1984 -> "nineteen eighty-four", 2005 -> "two thousand five", 1905 -> "nineteen oh five"."""
    if not 1000 <= n <= 2999 or 2000 <= n <= 2009 or n % 1000 == 0:
        return cardinal(n)
    return pairs(n)


def pairs(n: int) -> str:
    """Read a four-digit number as two pairs: 4070 -> "forty seventy", 1905 -> "nineteen oh five"."""
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{cardinal(hi)} hundred"
    if lo < 10:
        return f"{cardinal(hi)} oh {ONES[lo]}"
    return f"{cardinal(hi)} {cardinal(lo)}"


def plural(words: str) -> str:
    """Pluralise the last word: "nineteen ninety" -> "nineteen nineties", "third" -> "thirds"."""
    if words.endswith("y"):
        return words[:-1] + "ies"
    if words.endswith(("x", "s")):
        return words + "es"
    return words + "s"


def digits(s: str) -> str:
    """Read digit by digit: "007" -> "zero zero seven"."""
    return " ".join(ONES[int(c)] for c in s if c.isdigit())


def number(s: str) -> str:
    """Any written number: "-1,234.05" -> "minus one thousand two hundred thirty-four point zero five"."""
    s = s.replace(",", "").replace("−", "-")
    sign = ""
    if s.startswith("-"):
        sign, s = "minus ", s[1:]
    whole, _, frac = s.partition(".")
    if len(whole) > 1 and whole.startswith("0") or len(whole) > 15:
        spoken = digits(whole)
    else:
        spoken = cardinal(int(whole or "0"))
    if frac:
        spoken += " point " + digits(frac)
    return sign + spoken


# --- Rules ----------------------------------------------------------------------------------

MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
]
MONTH_RE = (
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)


def _month(name: str) -> str:
    return next(m for m in MONTHS if m.lower().startswith(name.lower()[:3]))


NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
#: A number not glued to a preceding word, decimal point or range ("5-10": the 10 is not "-10").
NUM_START = r"(?<![\w.,])(?<!\d-)"
#: An optional sign, a number and an optional range end: "-5", "3.5", "5-10".
VALUE = rf"(-)?({NUM})(?:\s?-\s?({NUM}))?"


def value(sign: str | None, a: str, b: str | None) -> str:
    """Speak a matched VALUE: "minus five", "five to ten"."""
    return ("minus " if sign else "") + number(a) + (f" to {number(b)}" if b else "")

CURRENCIES = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
    "¥": ("yen", "yen", "sen", "sen"),
}
SCALE_WORDS = {
    "k": "thousand", "K": "thousand", "thousand": "thousand",
    "m": "million", "M": "million", "mn": "million", "million": "million",
    "b": "billion", "B": "billion", "bn": "billion", "billion": "billion",
    "t": "trillion", "T": "trillion", "tn": "trillion", "trillion": "trillion",
}
SCALE_RE = r"(?:\s?(thousand|million|billion|trillion)|(k|K|M|mn|m|B|bn|b|T|tn|t))(?![\w])"

UNITS = {
    "km/h": ("kilometer per hour", "kilometers per hour"),
    "m/s": ("meter per second", "meters per second"),
    "mph": ("mile per hour", "miles per hour"),
    "kph": ("kilometer per hour", "kilometers per hour"),
    "km": ("kilometer", "kilometers"),
    "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"),
    "m": ("meter", "meters"),
    "mi": ("mile", "miles"),
    "ft": ("foot", "feet"),
    "kg": ("kilogram", "kilograms"),
    "mg": ("milligram", "milligrams"),
    "g": ("gram", "grams"),
    "lbs": ("pound", "pounds"),
    "lb": ("pound", "pounds"),
    "oz": ("ounce", "ounces"),
    "tbsp": ("tablespoon", "tablespoons"),
    "tsp": ("teaspoon", "teaspoons"),
    "ml": ("milliliter", "milliliters"),
    "mL": ("milliliter", "milliliters"),
    "L": ("liter", "liters"),
    "GHz": ("gigahertz", "gigahertz"),
    "MHz": ("megahertz", "megahertz"),
    "kHz": ("kilohertz", "kilohertz"),
    "Hz": ("hertz", "hertz"),
    "TB": ("terabyte", "terabytes"),
    "GB": ("gigabyte", "gigabytes"),
    "MB": ("megabyte", "megabytes"),
    "KB": ("kilobyte", "kilobytes"),
    "kB": ("kilobyte", "kilobytes"),
    "ms": ("millisecond", "milliseconds"),
    "sec": ("second", "seconds"),
    "min": ("minute", "minutes"),
    "hrs": ("hour", "hours"),
    "hr": ("hour", "hours"),
    "kW": ("kilowatt", "kilowatts"),
    "W": ("watt", "watts"),
    "V": ("volt", "volts"),
    "mAh": ("milliamp hour", "milliamp hours"),
}

ABBREVIATIONS = [
    (r"\bMr\.", "Mister"),
    (r"\bMrs\.", "Missus"),
    (r"\bMs\.", "Miz"),
    (r"\bDr\.", "Doctor"),
    (r"\bProf\.", "Professor"),
    (r"\bJr\.", "Junior"),
    (r"\bSr\.", "Senior"),
    (r"\bSt\.(?=\s+[A-Z])", "Saint"),
    (r"\bSt\.(?=\s*$)", "Street."),  # the abbreviation's dot also ends the sentence
    (r"\bSt\.", "Street"),
    (r"\bAve\.", "Avenue"),
    (r"\bvs\.?(?=\s)", "versus"),
    (r"\be\.g\.,?", "for example,"),
    (r"\bi\.e\.,?", "that is,"),
    (r"\betc\.(?=\s*[,;:)])", "et cetera"),
    (r"\betc\.", "et cetera."),
    (r"\bapprox\.", "approximately"),
    (r"\b[Nn]o\.\s?(?=\d)", "number "),
    (r"#(?=\d)", "number "),
]


def _sub(pattern: str, repl: Callable[[re.Match[str]], str], text: str, flags: int = 0) -> str:
    return re.sub(pattern, repl, text, flags=flags)


def _markdown(text: str) -> str:
    """Strip formatting an LLM reply may carry: links, emphasis, code ticks, headings, bullets."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(\*\*|__|`+)", "", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+)(?<!\s)\*(?![\w*])", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", text, flags=re.M)
    return text


UNICODE = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "−": "-",
    " ": " ", "…": "...", "½": " 1/2", "¼": " 1/4", "¾": " 3/4",
}


def _unicode(text: str) -> str:
    for a, b in UNICODE.items():
        text = text.replace(a, b)
    text = re.sub(r"\s*[—]\s*", ", ", text)  # em dash: a pause
    return re.sub(r"(?<=\d)–(?=\d)", "-", text).replace("–", ", ")


def _spell_address(s: str) -> str:
    for sym, word in (("@", " at "), (".", " dot "), ("/", " slash "), ("-", " dash "),
                      ("_", " underscore "), (":", " colon ")):
        s = s.replace(sym, word)
    return re.sub(r"\s+", " ", s).strip()


def _addresses(text: str) -> str:
    def url(m: re.Match[str]) -> str:
        core = re.sub(r"^https?://(www\.)?", "", m.group(1)).rstrip("/")
        return _spell_address(core) + m.group(2)

    text = _sub(r"(https?://[^\s]+?)([.,;:!?)]*(?:\s|$))", url, text)
    text = _sub(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", lambda m: _spell_address(m.group()), text)
    tlds = r"com|org|net|io|ai|dev|edu|gov|app|co\.uk"
    return _sub(rf"\b(?:www\.)?[\w-]+(?:\.[\w-]+)*\.(?:{tlds})\b(?:/[\w/-]*)?",
                lambda m: _spell_address(re.sub(r"^www\.", "", m.group())), text)


def _abbreviations(text: str) -> str:
    for pattern, word in ABBREVIATIONS:
        text = re.sub(pattern, word, text)
    return text


def _currency(text: str) -> str:
    def money(m: re.Match[str]) -> str:
        sign, sym, amount, scale_word, scale_abbr = m.groups()
        one, many, cent, cents = CURRENCIES[sym]
        prefix = "minus " if sign else ""
        scale = SCALE_WORDS.get(scale_word or scale_abbr or "")
        if scale:
            return f"{prefix}{number(amount)} {scale} {many}"
        whole, _, frac = amount.replace(",", "").partition(".")
        if len(frac) > 2:
            return f"{prefix}{number(amount)} {many}"
        units, sub = int(whole or "0"), int(frac.ljust(2, "0")) if frac else 0
        parts = []
        if units or not sub:
            parts.append(f"{cardinal(units)} {one if units == 1 else many}")
        if sub:
            parts.append(f"{cardinal(sub)} {cent if sub == 1 else cents}")
        return prefix + " and ".join(parts)

    return _sub(rf"(-)?([$£€¥])\s?({NUM})(?:{SCALE_RE})?", money, text)


def _dates(text: str) -> str:
    def spoken(month: int, day: int, yr: str | None) -> str | None:
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        out = f"{MONTHS[month - 1]} {ordinal(day)}"
        return out + (f", {year(int(yr))}" if yr else "")

    def iso(m: re.Match[str]) -> str:
        return spoken(int(m.group(2)), int(m.group(3)), m.group(1)) or m.group()

    def slash(m: re.Match[str]) -> str:
        yr = m.group(3)
        if len(yr) == 2:
            yr = str((2000 if int(yr) < 50 else 1900) + int(yr))
        return spoken(int(m.group(1)), int(m.group(2)), yr) or m.group()

    def month_day(m: re.Match[str]) -> str:
        out = f"{_month(m.group(1))} {ordinal(int(m.group(2)))}"
        return out + (f", {year(int(m.group(3)))}" if m.group(3) else "")

    def day_month(m: re.Match[str]) -> str:
        out = f"the {ordinal(int(m.group(1)))} of {_month(m.group(2))}"
        return out + (f", {year(int(m.group(3)))}" if m.group(3) else "")

    text = _sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", iso, text)
    text = _sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", slash, text)
    text = _sub(rf"\b{MONTH_RE}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}})\b)?",
                month_day, text)
    return _sub(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH_RE}\.?(?:,?\s+(\d{{4}})\b)?",
                day_month, text)


AMPM = r"([AaPp])\.?\s?[Mm]\.?(?![\w])"


def _times(text: str) -> str:
    def clock(m: re.Match[str]) -> str:
        h, mins, ap = int(m.group(1)), int(m.group(2)), m.group(3)
        if h > 23 or mins > 59:
            return m.group()
        if mins == 0:
            spoken = cardinal(h) + ("" if ap else " o'clock")
        elif mins < 10:
            spoken = f"{cardinal(h)} oh {ONES[mins]}"
        else:
            spoken = f"{cardinal(h)} {cardinal(mins)}"
        return spoken + (f" {ap.upper()} M" if ap else "")

    text = _sub(rf"\b(\d{{1,2}}):(\d{{2}})(?!:?\d)(?:\s?{AMPM})?", clock, text)
    return _sub(rf"\b(\d{{1,2}})\s?{AMPM}",
                lambda m: f"{cardinal(int(m.group(1)))} {m.group(2).upper()} M", text)


def _special_numbers(text: str) -> str:
    text = re.sub(r"\b24/7\b", "twenty-four seven", text)
    phone = r"(?<![\d-])(?:\(?(\d{3})\)?[\s.-])?(\d{3})-(\d{4})(?![\d-])"
    text = _sub(phone, lambda m: ", ".join(digits(g) for g in m.groups() if g), text)
    # Model numbers after a capitalised name read in pairs: "RTX 4070" -> "forty seventy".
    return _sub(r"\b([A-Z]{2,}\w*)\s([1-9]\d{3})\b",
                lambda m: f"{m.group(1)} {pairs(int(m.group(2)))}", text)


def _percent_and_units(text: str) -> str:
    text = _sub(rf"{NUM_START}{VALUE}\s?%", lambda m: f"{value(*m.groups())} percent", text)

    def single(m: re.Match[str]) -> bool:
        return m.group(2) == "1" and not m.group(1) and not m.group(3)

    def degrees(m: re.Match[str]) -> str:
        scale = {"C": " Celsius", "F": " Fahrenheit", "K": " Kelvin"}.get(m.group(4) or "", "")
        return f"{value(*m.groups()[:3])} {'degree' if single(m) else 'degrees'}{scale}"

    text = _sub(rf"{NUM_START}{VALUE}\s?°\s?([CFK](?![\w]))?", degrees, text)

    names = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))

    def unit(m: re.Match[str]) -> str:
        one, many = UNITS[m.group(4)]
        return f"{value(*m.groups()[:3])} {one if single(m) else many}"

    return _sub(rf"{NUM_START}{VALUE}\s?({names})(?![\w/])", unit, text)


def _scaled(text: str) -> str:
    """Bare scale suffixes: "5M users" -> "five million users", "2.5k" -> "two point five thousand"."""
    return _sub(rf"{NUM_START}(\d+(?:\.\d+)?)(k|K|M|mn|B|bn|T|tn)(?![\w])",
                lambda m: f"{number(m.group(1))} {SCALE_WORDS[m.group(2)]}", text)


YEAR_CONTEXT = r"(?:in|since|from|by|until|till|before|after|of|circa|during|around|year|early|late)"


def _years(text: str) -> str:
    def decade(m: re.Match[str]) -> str:
        if m.group(1):
            return plural(year(int(m.group(1))))
        return plural(TENS[int(m.group(2)) // 10])

    text = _sub(r"\b(\d{3}0)'?s\b|(?:'|\b)([1-9]0)'?s\b", decade, text)
    text = _sub(rf"\b({YEAR_CONTEXT})\s+(1[1-9]\d\d|20\d\d)\b",
                lambda m: f"{m.group(1)} {year(int(m.group(2)))}", text, flags=re.I)

    def year_range(m: re.Match[str]) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        if 1500 <= a < b <= 2099:
            return f"{year(a)} to {year(b)}"
        return m.group()

    return _sub(r"\b(\d{4})\s?-\s?(\d{4})\b", year_range, text)


def _ordinals(text: str) -> str:
    return _sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), text)


def _fractions(text: str) -> str:
    def fraction(m: re.Match[str]) -> str:
        n, d = int(m.group(1)), int(m.group(2))
        if d == 0 or d > 16:
            return f"{cardinal(n)} over {cardinal(d)}"
        name = {2: "half", 4: "quarter"}.get(d) or ordinal(d)
        if n != 1:
            name = "halves" if d == 2 else plural(name)
        return f"{cardinal(n)} {name}"

    text = _sub(r"(\d)\s+(?=\d+/\d+\b)", lambda m: f"{m.group(1)} and ", text)  # 2 1/2
    return _sub(rf"{NUM_START}(\d{{1,3}})/(\d{{1,3}})\b(?!/)", fraction, text)


def _ranges(text: str) -> str:
    return _sub(rf"{NUM_START}({NUM})\s?-\s?({NUM})(?!\w)(?!\.\d)",
                lambda m: f"{number(m.group(1))} to {number(m.group(2))}", text)


def _numbers(text: str) -> str:
    def spoken(m: re.Match[str]) -> str:
        glued = m.string[m.end() : m.end() + 1].isalpha()  # "3D" -> "three D"
        return ("minus " if m.group(1) else "") + number(m.group(2)) + (" " if glued else "")

    # A minus sign only counts at the start or after a space or bracket ("COVID-19" has none).
    return _sub(rf"(?:(?<![^\s(])(-))?{NUM_START}({NUM})(?!\d)", spoken, text)


def _symbols(text: str) -> str:
    text = re.sub(r"\band/or\b", "and or", text)
    text = re.sub(r"(?<=[A-Za-z])/(?=[A-Za-z])", " or ", text)
    text = re.sub(r"~\s?(?=\w)", "about ", text)
    for sym, word in (("%", " percent "), ("&", " and "), ("+", " plus "), ("=", " equals "), ("×", " times "),
                      ("@", " at "), ("°", " degrees ")):
        text = text.replace(sym, word)
    text = re.sub(r"[*_#|<>\[\]{}\\^]", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    return re.sub(r"\s+", " ", text).strip()


RULES: list[Callable[[str], str]] = [
    _markdown, _unicode, _addresses, _abbreviations, _currency, _dates, _times,
    _special_numbers, _percent_and_units, _scaled, _years, _ordinals, _fractions, _ranges,
    _numbers, _symbols,
]


def normalize(text: str) -> str:
    """Rewrite `text` into plain speakable words, keeping sentence punctuation for prosody."""
    for rule in RULES:
        text = rule(text)
    return text
