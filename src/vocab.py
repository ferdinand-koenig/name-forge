# ------------------------------
# Business types
# ------------------------------
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

# ------------------------------
# Generic business types (formerly used only in unsafe generator)
# Used to diversify safe examples
# ------------------------------
generic_business_types = [
    "website",
    "platform",
    "service",
    "store",
    "marketplace",
]

# ------------------------------
# Adjectives
# ------------------------------
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

# Rare / unusual adjectives for creative variation
rare_adjectives = [
    "cosmic",
    "quantum",
    "mythic",
    "vivid",
    "atomic",
    "stellar",
    "radiant",
    "luminous",
    "ethereal",
    "dynamic",
]

# ------------------------------
# Nouns
# ------------------------------
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

# Rare / unusual nouns for creative variation
rare_nouns = [
    "haven",
    "forge",
    "nexus",
    "sphere",
    "vault",
    "deck",
    "realm",
    "sanctum",
    "tower",
    "arena",
]

# ------------------------------
# Top-level domains
# ------------------------------
tlds = [".com", ".net", ".org", ".io", ".co"]

# ------------------------------
# Locations (for medium/complex descriptions)
# ------------------------------
locations = [
    "downtown",
    "suburban area",
    "city center",
    "neighborhood",
    "near the river",
]

# ------------------------------
# Purposes (for complex descriptions)
# ------------------------------
purposes = [
    "for busy professionals",
    "with focus on sustainability",
    "offering premium services",
    "specializing in community events",
]

# ------------------------------
# Suffixes for domain names
# ------------------------------
suffixes = ["hub", "lab", "works", "studio", "corner", "solutions"]

# ------------------------------
# Edge case descriptions (realistic but challenging)
# ------------------------------
edge_descriptions = [
    "premium eco-friendly coffee and bakery shop in downtown",
    "quantum tech startup with cosmic lab solutions",
    "bright & cozy clothing boutique",
    "vivid vivid hub for creative creators",
    "extremely long travel agency offering global "
    + "adventure and premium services in city center",
    "tiny bookstore",
    "mythic nexus for innovative ideas",
    "atomic studio for fresh and vivid projects",
    "stellar vault of creative educational platform",
    "cosmic forge of premium online stores",
]

# ------------------------------
# Unsafe / forbidden categories
# ------------------------------
unsafe_categories = {
    "adult": [
        "adult content",
        "pornographic",
        "erotic",
        "sex",
        "nude",
        "porn",
    ],
    "gambling": ["illegal gambling", "betting", "casino", "lottery"],
    "piracy": ["pirated", "copyright", "streaming"],
    "violence": ["weapons", "arms", "violent"],
    "drugs": ["drugs", "narcotics", "marijuana"],
}
