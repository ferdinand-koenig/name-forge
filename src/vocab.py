# src/vocab.py

# ------------------------------
# Business types keywords for relevant domain generation
# ------------------------------
business_type_keywords = {
    "bookstore": [
        "books",
        "library",
        "store",
        "reads",
        "vault",
        "corner",
        "shelf",
    ],
    "online store": [
        "shop",
        "store",
        "market",
        "corner",
        "emporium",
        "bazaar",
    ],
    "clothing boutique": [
        "fashion",
        "style",
        "boutique",
        "threads",
        "wardrobe",
        "apparel",
    ],
    "fitness studio": ["fit", "gym", "studio", "hub", "training", "wellness"],
    "travel agency": [
        "travel",
        "tours",
        "trips",
        "journey",
        "voyage",
        "expedition",
    ],
    "restaurant": [
        "dine",
        "eatery",
        "kitchen",
        "bistro",
        "grill",
        "table",
        "cafe",
    ],
    "tech startup": [
        "tech",
        "lab",
        "hub",
        "works",
        "innovate",
        "solutions",
        "platform",
    ],
    "law firm": ["law", "legal", "firm", "counsel", "advocate", "partners"],
    "medical clinic": [
        "health",
        "clinic",
        "care",
        "wellness",
        "med",
        "center",
    ],
    "coffee shop": ["coffee", "brew", "cafe", "roasters", "beans", "corner"],
    "art gallery": ["gallery", "art", "studio", "exhibit", "canvas", "works"],
    "cosmetics brand": [
        "beauty",
        "cosmetics",
        "glow",
        "vanity",
        "studio",
        "essence",
    ],
    "music school": [
        "music",
        "academy",
        "studio",
        "notes",
        "melody",
        "harmony",
    ],
    "software company": [
        "soft",
        "solutions",
        "tech",
        "labs",
        "platform",
        "systems",
    ],
    "consulting firm": [
        "consult",
        "advisory",
        "partners",
        "solutions",
        "group",
        "hub",
    ],
    "pet store": ["pets", "paw", "store", "petcare", "corner", "buddy"],
    "bakery": ["bake", "bakery", "oven", "bread", "sweet", "corner"],
    "toy store": ["toys", "play", "corner", "fun", "games", "hub"],
    "garden center": [
        "garden",
        "plants",
        "nursery",
        "green",
        "flora",
        "corner",
    ],
    "fitness apparel": [
        "fit",
        "active",
        "gear",
        "threads",
        "style",
        "performance",
    ],
    "cafe": ["coffee", "brew", "cafe", "corner", "beans"],
    "educational platform": ["learn", "academy", "platform", "study", "hub"],
}

# ------------------------------
# Business types list
# ------------------------------
business_types = list(business_type_keywords.keys())

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
