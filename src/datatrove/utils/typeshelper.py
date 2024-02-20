from dataclasses import dataclass


@dataclass
class NiceRepr:
    emoji: str
    name: str

    def __post_init__(self):
        self.name = self.name.capitalize()

    def get_name(self):
        return f"---> {self.emoji} {self.name}\n"


class Languages:
    english = "en"
    spanish = "es"
    portuguese = "pt"
    italian = "it"
    french = "fr"
    swedish = "sv"
    romanian = "ro"
    german = "de"
    latin = "la"
    czech = "cs"
    danish = "da"
    estonian = "et"
    finnish = "fi"
    greek = "el"
    malayalam = "ml"
    norwegian = "no"
    polish = "pl"
    portuguese = "pt"
    russian = "ru"
    slovenian = "sl"
    swedish = "sv"
    turkish = "tr"
    dutch = "nl"
    chinese = "zh"
    japanese = "ja"


class StatHints:
    total = "total"
    dropped = "dropped"
    forwarded = "forwarded"
