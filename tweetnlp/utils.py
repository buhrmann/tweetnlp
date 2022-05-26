from enum import Enum


class Task(Enum):
    emoji = "emoji"
    emotion = "emotion"
    hate = "hate"
    irony = "irony"
    offensive = "offensive"
    sentiment = "sentiment"  # pred and label size mismatch?
    stance = "stance"


class StanceTopic(Enum):
    abortion = "abortion"
    atheism = "atheism"
    climate = "climate"
    feminist = "feminist"
    hillary = "hillary"


class Split(Enum):
    train = "train"
    test = "test"
    val = "val"


def preprocess(text: str) -> str:
    """Same preprocessing used as in pretrained models."""
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)
