from __future__ import annotations

import os
from pathlib import Path
from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.9))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 800))

BASE_DIR = Path(__file__).parent / "synthetic_data"
RELEVANT_DIR = BASE_DIR / "relevant"
IRRELEVANT_DIR = BASE_DIR / "irrelevant"


ARTISTS: List[str] = [
    "Lady Gaga", "Coldplay", "BTS", "Adele", "Bad Bunny", "Taylor Swift",
    "Metallica", "Billie Eilish", "Ed Sheeran", "Dua Lipa", "Harry Styles",
    "Rammstein", "Rihanna", "Drake", "Bruno Mars", "The Weeknd",
    "Imagine Dragons", "Kendrick Lamar", "Arctic Monkeys", "BLACKPINK",
    "Foo Fighters", "Shakira", "Post Malone", "Kacey Musgraves",
    "J Balvin", "Lizzo", "Red Hot Chili Peppers", "Muse", "Ariana Grande",
    "Jay‑Z", "Kings of Leon", "Green Day", "Hozier", "SZA", "Måneskin",
    "Karol G", "John Mayer", "Paramore", "Sam Smith", "Tame Impala",
]


NON_CONCERT_TOPICS: List[str] = [
    "Sourdough bread baking techniques", "Basics of quantum computing",
    "History of the Silk Road", "Indoor hydroponic gardening guide",
    "DIY home solar panel installation", "Caring for freshwater aquariums",
    "Introduction to astrophotography", "Beginner yoga routine",
    "Making artisanal cheese at home", "Urban beekeeping handbook",
    "Classic French pastry recipes", "Restoring vintage motorcycles",
    "Wildlife photography in Patagonia", "Learning the Go programming language",
    "Building a backyard wood‑fired pizza oven", "Understanding blockchain for supply chains",
    "Mediterranean diet meal prep", "Brewing specialty coffee at home",
    "Basics of 3D printing", "Creating a personal knowledge management system",
    "History of Armenian khachkars", "Introduction to minimalism lifestyle",
    "Bird‑watching in Yerevan parks", "Guide to composting for city dwellers",
    "Beginner’s guide to calligraphy", "Designing low‑budget tiny houses",
    "Exploring dark matter theories", "How to start a podcast",
    "Mindfulness meditation for stress", "Training for a half‑marathon",
    "Culinary herbs and their uses", "Introduction to screenwriting",
    "Hiking the Appalachian Trail", "Photography lighting setups",
    "Basic car engine maintenance", "History of the Armenian Genocide",
    "Learning to play the ukulele", "Wine tasting for beginners",
    "Crafting with recycled materials", "Data visualization with Python",
]


assert len(ARTISTS) == 40 and len(NON_CONCERT_TOPICS) == 40, "Need 40 of each"

llm = ChatOpenAI(model_name=LLM_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

concert_prompt = ChatPromptTemplate.from_template(
    """ You are a tour-manager assistant. Give a detailed and realistic 2025-2026 world-tour plan for
     the artist {artist}. include:
        - Tour name
        - 6-10 cities with venues and exact dates
        - Key logistical notes (e.g. stage design, local regulations)
        - Ticket sale launch dates
     Write in plain text, ~250‑300 words
    """
)

irrelevant_prompt = ChatPromptTemplate.from_template(
    """Write an informative, engaging ~250‑300‑word article about {topic}.
    Do NOT mention music, concerts, or tours."""
)


def generate_doc(prompt: ChatPromptTemplate, subject: str) -> str:
    """Generate a document given a prompt template and a subject string"""
    messages: List[BaseMessage] = prompt.format_messages(**{prompt.input_variables[0]: subject})
    response = llm.invoke(messages)
    return response.content.strip()

def write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"Saved {path.relative_to(BASE_DIR)}")


def main() -> None:
    RELEVANT_DIR.mkdir(parents=True, exist_ok=True)
    IRRELEVANT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating relevant concert‑tour documents")
    for idx, artist in enumerate(ARTISTS, start = 1):
        doc_text = generate_doc(concert_prompt, artist)
        filename = f"{idx:02d}_{artist.replace(' ', '_')}.txt"
        write_file(RELEVANT_DIR / filename, doc_text)
    print("\n")
    print("Generating irrelevant documents")
    for idx, topic in enumerate(NON_CONCERT_TOPICS, start = 1):
        doc_text = generate_doc(irrelevant_prompt, topic)
        sanitized = "_".join(topic.lower().split()[:4])
        filename = f"{idx:02d}_{sanitized}.txt"
        write_file(IRRELEVANT_DIR / filename, doc_text)

    print("done generating synthetic data")

if __name__ == "__main__":
    main()