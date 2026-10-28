"""Text normalisation for speech: written text becomes the words a person would say."""

from __future__ import annotations

import re

import pytest

from edith.text import cardinal, normalize, number, ordinal, pairs, year


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "zero"),
        (7, "seven"),
        (13, "thirteen"),
        (40, "forty"),
        (42, "forty-two"),
        (100, "one hundred"),
        (101, "one hundred one"),
        (999, "nine hundred ninety-nine"),
        (1000, "one thousand"),
        (1_000_001, "one million one"),
        (2_500_000_000, "two billion five hundred million"),
        (-15, "minus fifteen"),
    ],
)
def test_cardinal(n: int, words: str) -> None:
    assert cardinal(n) == words


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (1, "first"), (2, "second"), (3, "third"), (5, "fifth"), (8, "eighth"), (9, "ninth"),
        (12, "twelfth"), (20, "twentieth"), (21, "twenty-first"), (100, "one hundredth"),
        (111, "one hundred eleventh"),
    ],
)
def test_ordinal(n: int, words: str) -> None:
    assert ordinal(n) == words


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (1984, "nineteen eighty-four"),
        (1900, "nineteen hundred"),
        (1905, "nineteen oh five"),
        (2000, "two thousand"),
        (2005, "two thousand five"),
        (2010, "twenty ten"),
        (2024, "twenty twenty-four"),
        (1100, "eleven hundred"),
    ],
)
def test_year(n: int, words: str) -> None:
    assert year(n) == words


def test_pairs_reads_model_numbers() -> None:
    assert pairs(4070) == "forty seventy"


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("3.14", "three point one four"),
        ("1,000,000", "one million"),
        ("-2.5", "minus two point five"),
        ("007", "zero zero seven"),
        ("0.5", "zero point five"),
    ],
)
def test_number(written: str, spoken: str) -> None:
    assert number(written) == spoken


CASES = [
    # Currency
    ("It costs $12.50.", "It costs twelve dollars and fifty cents."),
    ("$1,234.56", "one thousand two hundred thirty-four dollars and fifty-six cents"),
    ("$1", "one dollar"),
    ("$0.99", "ninety-nine cents"),
    ("$5.01", "five dollars and one cent"),
    ("€20", "twenty euros"),
    ("£1.50", "one pound and fifty pence"),
    ("¥500", "five hundred yen"),
    ("$2.5M", "two point five million dollars"),
    ("$3 billion", "three billion dollars"),
    ("-$5", "minus five dollars"),
    # Dates
    ("2024-03-15", "March fifteenth, twenty twenty-four"),
    ("3/15/2024", "March fifteenth, twenty twenty-four"),
    ("3/15/24", "March fifteenth, twenty twenty-four"),
    ("Mar 5, 2025", "March fifth, twenty twenty-five"),
    ("on July 4th", "on July fourth"),
    ("15 March 2024", "the fifteenth of March, twenty twenty-four"),
    ("Sept 1", "September first"),
    # Times
    ("at 3:30 pm", "at three thirty P M"),
    ("at 10:00", "at ten o'clock"),
    ("at 9:05 a.m.", "at nine oh five A M"),
    ("at 14:45", "at fourteen forty-five"),
    ("9am", "nine A M"),
    ("7 PM", "seven P M"),
    # Percent, temperature, units
    ("up 12%", "up twelve percent"),
    ("-3.5%", "minus three point five percent"),
    ("5-10%", "five to ten percent"),
    ("20°C", "twenty degrees Celsius"),
    ("-5 °F", "minus five degrees Fahrenheit"),
    ("1°", "one degree"),
    ("100 km/h", "one hundred kilometers per hour"),
    ("1 km", "one kilometer"),
    ("1.0 km", "one point zero kilometers"),
    ("5-10 kg", "five to ten kilograms"),
    ("250 ms", "two hundred fifty milliseconds"),
    ("16 GB", "sixteen gigabytes"),
    ("60Hz", "sixty hertz"),
    ("2 tsp", "two teaspoons"),
    ("6 ft", "six feet"),
    # Scales and plain numbers
    ("5k followers", "five thousand followers"),
    ("1,500 people", "one thousand five hundred people"),
    ("-7 degrees", "minus seven degrees"),
    ("COVID-19", "COVID-nineteen"),
    ("in 3D", "in three D"),
    # Years and decades
    ("In 1984 it rained.", "In nineteen eighty-four it rained."),
    ("since 2010", "since twenty ten"),
    ("1500 people", "one thousand five hundred people"),
    ("the 1990s", "the nineteen nineties"),
    ("the '80s", "the eighties"),
    ("the 80s", "the eighties"),
    ("1990-2000", "nineteen ninety to two thousand"),
    # Ordinals, fractions, ranges
    ("the 23rd", "the twenty-third"),
    ("1/2 cup", "one half cup"),
    ("3/4", "three quarters"),
    ("2/3", "two thirds"),
    ("2 1/2", "two and one half"),
    ("½", "one half"),
    ("pages 10-20.", "pages ten to twenty."),
    ("open 24/7", "open twenty-four seven"),
    # Model numbers and phone numbers
    ("an RTX 4070", "an RTX forty seventy"),
    ("call 555-123-4567", "call five five five, one two three, four five six seven"),
    ("(555) 123-4567", "five five five, one two three, four five six seven"),
    # Abbreviations
    ("Dr. Lee", "Doctor Lee"),
    ("Mr. and Mrs. Smith", "Mister and Missus Smith"),
    ("St. Louis", "Saint Louis"),
    ("Main St.", "Main Street."),
    ("cats vs. dogs", "cats versus dogs"),
    ("e.g. apples", "for example, apples"),
    ("i.e. fruit", "that is, fruit"),
    ("apples, pears, etc.", "apples, pears, et cetera."),
    ("pears, etc., and", "pears, et cetera, and"),
    ("No. 5", "number five"),
    ("#3", "number three"),
    # Addresses
    ("https://www.example.com/docs.", "example dot com slash docs."),
    ("jo.doe@mail.org", "jo dot doe at mail dot org"),
    ("see github.io", "see github dot io"),
    # Symbols and punctuation
    ("salt & pepper", "salt and pepper"),
    ("2+2=4", "two plus two equals four"),
    ("cats/dogs", "cats or dogs"),
    ("and/or", "and or"),
    ("~5 of them", "about five of them"),
    ("Wait — what?", "Wait, what?"),
    ("“Hi,” she said…", '"Hi," she said...'),
    # Markdown from LLM replies
    ("**Bold** and *italic* and `code`", "Bold and italic and code"),
    ("[a link](http://x.y)", "a link"),
    ("# Title\n- one item", "Title one item"),
]


@pytest.mark.parametrize(("written", "spoken"), CASES)
def test_normalize(written: str, spoken: str) -> None:
    assert normalize(written) == spoken


def test_full_sentence() -> None:
    assert normalize(
        "Meet Dr. Lee at 3:30 pm on Mar 5, 2025 — bring $12.50 & a 2 kg bag."
    ) == (
        "Meet Doctor Lee at three thirty P M on March fifth, twenty twenty-five, bring twelve "
        "dollars and fifty cents and a two kilograms bag."
    )


def test_plain_text_is_unchanged() -> None:
    s = "Hello there. How are you today? I'm fine, thanks!"
    assert normalize(s) == s


@pytest.mark.parametrize("written", [c[0] for c in CASES])
def test_output_has_no_digits_or_currency_symbols(written: str) -> None:
    assert not re.search(r"[\d$€£¥%°&@#]", normalize(written))
