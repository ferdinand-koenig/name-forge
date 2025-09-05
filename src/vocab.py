# src/vocab.py

business_types = [
    "cafe",
    "restaurant",
    "tech startup",
    "online store",
    "clothing boutique",
    "law firm",
    "educational platform",
    "fitness studio",
    "travel agency",
    "bookstore",
]

adjectives = [
    "organic",
    "eco-friendly",
    "bright",
    "cozy",
    "modern",
    "smart",
    "fresh",
    "premium",
    "global",
    "innovative",
]

nouns = [
    "beans",
    "shop",
    "hub",
    "store",
    "lab",
    "studio",
    "solutions",
    "works",
    "spot",
    "corner",
]

tlds = [".com", ".net", ".org", ".io", ".co"]

# Edge case descriptions
edge_descriptions = [
    "business with unusual characters @$%^",
    "extremely long business name that goes on and on for testing purposes",
    "very short",
]

# Forbidden words for safety
FORBIDDEN_WORDS = ["adult", "porn", "nude", "illegal"]
