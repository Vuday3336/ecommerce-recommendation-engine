"""Catalogue taxonomy, brands and the complementarity graph.

This file is where the *learnable structure* of the dataset is declared. It is
data, not code, and it is deliberately explicit rather than randomly generated:
a recommender can only rediscover structure that was put there in the first
place (risk R-02), and structure sampled from a uniform prior is noise wearing
a taxonomy's clothes.

Three kinds of structure are encoded here:

1. **Hierarchy.** Departments contain categories contain subcategories. This is
   what category-affinity candidate generation and the diversity re-ranker
   operate on.
2. **Brand positioning.** Each brand sells into a limited set of departments at
   a consistent price tier. A user with a brand affinity therefore has a
   coherent, discoverable footprint across the catalogue.
3. **Complementarity.** Explicit pairs of subcategories that are bought
   together (running shoes and athletic socks; laptops and laptop bags). This
   is the ground truth that "frequently bought together" must recover, and it
   is the only reason FBT can be evaluated rather than merely computed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SubcategorySpec:
    """A leaf category: the grain products are assigned to."""

    name: str
    #: Multiplier on the department's base price. A 'Laptops' subcategory is
    #: expensive relative to 'Cables' even within the same department.
    price_multiplier: float
    #: Probability a purchase is repeated later. Consumables approach 1.0,
    #: durables approach 0. Drives repeat-purchase behaviour and the
    #: already-purchased business rule.
    repeat_rate: float
    #: Mean days between repeat purchases when a repeat happens.
    repeat_interval_days: float
    #: Attribute keys sampled for products in this subcategory.
    attributes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CategorySpec:
    name: str
    subcategories: tuple[SubcategorySpec, ...]


@dataclass(frozen=True, slots=True)
class DepartmentSpec:
    name: str
    #: Median product price in the department, before subcategory and brand
    #: multipliers. Prices are lognormal around this.
    base_price: float
    #: Seasonal peak as a day-of-year phase (0.0 = 1 January, 0.5 = early July)
    #: and amplitude. Winter apparel peaks in Q4; garden peaks in spring.
    season_phase: float
    season_amplitude: float
    categories: tuple[CategorySpec, ...]


def _sub(
    name: str,
    price_multiplier: float = 1.0,
    repeat_rate: float = 0.05,
    repeat_interval_days: float = 120.0,
    attributes: tuple[str, ...] = (),
) -> SubcategorySpec:
    return SubcategorySpec(
        name=name,
        price_multiplier=price_multiplier,
        repeat_rate=repeat_rate,
        repeat_interval_days=repeat_interval_days,
        attributes=attributes,
    )


COLOUR_ATTRS = ("colour", "material")
SIZE_ATTRS = ("colour", "size", "material")
TECH_ATTRS = ("colour", "capacity", "connectivity")


DEPARTMENTS: tuple[DepartmentSpec, ...] = (
    DepartmentSpec(
        name="Electronics",
        base_price=180.0,
        season_phase=0.92,  # peaks late November (Black Friday)
        season_amplitude=0.55,
        categories=(
            CategorySpec(
                "Computers",
                (
                    _sub("Laptops", 6.0, 0.02, 900, TECH_ATTRS),
                    _sub("Monitors", 2.2, 0.05, 700, TECH_ATTRS),
                    _sub("Keyboards", 0.6, 0.10, 500, TECH_ATTRS),
                    _sub("Mice", 0.4, 0.12, 450, TECH_ATTRS),
                    _sub("Laptop Bags", 0.5, 0.08, 600, COLOUR_ATTRS),
                ),
            ),
            CategorySpec(
                "Audio",
                (
                    _sub("Headphones", 1.2, 0.10, 500, TECH_ATTRS),
                    _sub("Earbuds", 0.9, 0.18, 350, TECH_ATTRS),
                    _sub("Speakers", 1.4, 0.06, 700, TECH_ATTRS),
                    _sub("Audio Cables", 0.15, 0.30, 200, TECH_ATTRS),
                ),
            ),
            CategorySpec(
                "Mobile",
                (
                    _sub("Smartphones", 5.0, 0.03, 800, TECH_ATTRS),
                    _sub("Phone Cases", 0.18, 0.35, 180, COLOUR_ATTRS),
                    _sub("Chargers", 0.22, 0.32, 220, TECH_ATTRS),
                    _sub("Screen Protectors", 0.12, 0.45, 150, COLOUR_ATTRS),
                ),
            ),
        ),
    ),
    DepartmentSpec(
        name="Apparel",
        base_price=55.0,
        season_phase=0.80,
        season_amplitude=0.45,
        categories=(
            CategorySpec(
                "Footwear",
                (
                    _sub("Running Shoes", 2.0, 0.22, 240, SIZE_ATTRS),
                    _sub("Trail Shoes", 2.2, 0.15, 300, SIZE_ATTRS),
                    _sub("Sneakers", 1.6, 0.20, 260, SIZE_ATTRS),
                    _sub("Boots", 2.4, 0.10, 400, SIZE_ATTRS),
                    _sub("Sandals", 0.9, 0.18, 320, SIZE_ATTRS),
                ),
            ),
            CategorySpec(
                "Activewear",
                (
                    _sub("Athletic Socks", 0.25, 0.55, 90, SIZE_ATTRS),
                    _sub("Running Shorts", 0.7, 0.30, 180, SIZE_ATTRS),
                    _sub("Sports Bras", 0.6, 0.35, 160, SIZE_ATTRS),
                    _sub("Training Tops", 0.8, 0.30, 170, SIZE_ATTRS),
                ),
            ),
            CategorySpec(
                "Outerwear",
                (
                    _sub("Rain Jackets", 2.0, 0.08, 500, SIZE_ATTRS),
                    _sub("Winter Coats", 3.2, 0.06, 600, SIZE_ATTRS),
                    _sub("Fleece", 1.3, 0.15, 300, SIZE_ATTRS),
                ),
            ),
        ),
    ),
    DepartmentSpec(
        name="Home",
        base_price=70.0,
        season_phase=0.05,
        season_amplitude=0.30,
        categories=(
            CategorySpec(
                "Kitchen",
                (
                    _sub("Cookware", 1.8, 0.08, 600, COLOUR_ATTRS),
                    _sub("Coffee Makers", 1.6, 0.05, 800, COLOUR_ATTRS),
                    _sub("Coffee Beans", 0.22, 0.75, 30, ("origin", "roast")),
                    _sub("Knives", 1.2, 0.10, 500, COLOUR_ATTRS),
                ),
            ),
            CategorySpec(
                "Bedding",
                (
                    _sub("Sheet Sets", 1.0, 0.12, 400, SIZE_ATTRS),
                    _sub("Pillows", 0.6, 0.20, 300, SIZE_ATTRS),
                    _sub("Duvets", 1.5, 0.08, 600, SIZE_ATTRS),
                ),
            ),
            CategorySpec(
                "Storage",
                (
                    _sub("Shelving", 1.4, 0.10, 500, COLOUR_ATTRS),
                    _sub("Storage Boxes", 0.35, 0.28, 220, COLOUR_ATTRS),
                ),
            ),
        ),
    ),
    DepartmentSpec(
        name="Beauty",
        base_price=32.0,
        season_phase=0.95,
        season_amplitude=0.35,
        categories=(
            CategorySpec(
                "Skincare",
                (
                    _sub("Moisturisers", 1.0, 0.70, 45, ("skin_type", "volume")),
                    _sub("Cleansers", 0.8, 0.72, 40, ("skin_type", "volume")),
                    _sub("Serums", 1.6, 0.60, 55, ("skin_type", "volume")),
                    _sub("Sunscreen", 0.7, 0.65, 50, ("spf", "volume")),
                ),
            ),
            CategorySpec(
                "Haircare",
                (
                    _sub("Shampoo", 0.6, 0.78, 35, ("hair_type", "volume")),
                    _sub("Conditioner", 0.6, 0.76, 35, ("hair_type", "volume")),
                    _sub("Hair Tools", 2.2, 0.05, 700, COLOUR_ATTRS),
                ),
            ),
        ),
    ),
    DepartmentSpec(
        name="Sports",
        base_price=85.0,
        season_phase=0.35,
        season_amplitude=0.50,
        categories=(
            CategorySpec(
                "Fitness",
                (
                    _sub("Yoga Mats", 0.5, 0.10, 500, COLOUR_ATTRS),
                    _sub("Dumbbells", 0.9, 0.12, 450, COLOUR_ATTRS),
                    _sub("Resistance Bands", 0.25, 0.25, 250, COLOUR_ATTRS),
                    _sub("Foam Rollers", 0.35, 0.15, 400, COLOUR_ATTRS),
                ),
            ),
            CategorySpec(
                "Outdoor",
                (
                    _sub("Tents", 3.0, 0.04, 900, COLOUR_ATTRS),
                    _sub("Sleeping Bags", 1.6, 0.05, 800, COLOUR_ATTRS),
                    _sub("Backpacks", 1.3, 0.08, 600, COLOUR_ATTRS),
                    _sub("Water Bottles", 0.25, 0.35, 200, COLOUR_ATTRS),
                ),
            ),
            CategorySpec(
                "Cycling",
                (
                    _sub("Bike Helmets", 1.0, 0.06, 700, SIZE_ATTRS),
                    _sub("Bike Lights", 0.4, 0.20, 300, TECH_ATTRS),
                    _sub("Cycling Jerseys", 0.9, 0.20, 280, SIZE_ATTRS),
                ),
            ),
        ),
    ),
    DepartmentSpec(
        name="Books",
        base_price=18.0,
        season_phase=0.90,
        season_amplitude=0.40,
        categories=(
            CategorySpec(
                "Fiction",
                (
                    _sub("Literary Fiction", 1.0, 0.45, 60, ("format", "language")),
                    _sub("Mystery", 0.95, 0.55, 45, ("format", "language")),
                    _sub("Science Fiction", 1.0, 0.55, 45, ("format", "language")),
                ),
            ),
            CategorySpec(
                "Non-Fiction",
                (
                    _sub("Business", 1.3, 0.35, 90, ("format", "language")),
                    _sub("Cookbooks", 1.4, 0.30, 120, ("format", "language")),
                    _sub("Science", 1.2, 0.35, 100, ("format", "language")),
                ),
            ),
        ),
    ),
    DepartmentSpec(
        name="Toys",
        base_price=28.0,
        season_phase=0.93,
        season_amplitude=0.75,  # strongly seasonal
        categories=(
            CategorySpec(
                "Games",
                (
                    _sub("Board Games", 1.4, 0.25, 180, ("players", "age_range")),
                    _sub("Puzzles", 0.7, 0.35, 120, ("pieces", "age_range")),
                    _sub("Card Games", 0.4, 0.40, 100, ("players", "age_range")),
                ),
            ),
            CategorySpec(
                "Building",
                (
                    _sub("Building Blocks", 1.6, 0.30, 150, ("pieces", "age_range")),
                    _sub("Model Kits", 1.2, 0.25, 200, ("pieces", "age_range")),
                ),
            ),
        ),
    ),
    DepartmentSpec(
        name="Pet",
        base_price=35.0,
        season_phase=0.45,
        season_amplitude=0.20,
        categories=(
            CategorySpec(
                "Dog",
                (
                    _sub("Dog Food", 0.9, 0.85, 28, ("weight", "life_stage")),
                    _sub("Dog Toys", 0.35, 0.40, 90, COLOUR_ATTRS),
                    _sub("Leashes", 0.4, 0.15, 400, COLOUR_ATTRS),
                ),
            ),
            CategorySpec(
                "Cat",
                (
                    _sub("Cat Food", 0.8, 0.86, 26, ("weight", "life_stage")),
                    _sub("Cat Litter", 0.5, 0.88, 24, ("weight",)),
                    _sub("Cat Toys", 0.3, 0.42, 90, COLOUR_ATTRS),
                ),
            ),
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class BrandSpec:
    name: str
    #: One of budget / mid / premium / luxury. Multiplies product price and
    #: shifts the quality (rating) distribution.
    tier: str
    #: Departments this brand sells into. Keeping brands narrow is what makes
    #: brand affinity a usable signal: a brand present in every department
    #: carries no information about taste.
    departments: tuple[str, ...]


BRANDS: tuple[BrandSpec, ...] = (
    # Electronics
    BrandSpec("Nordlys", "premium", ("Electronics",)),
    BrandSpec("Voltara", "mid", ("Electronics",)),
    BrandSpec("PixelForge", "premium", ("Electronics",)),
    BrandSpec("Kitto", "budget", ("Electronics",)),
    BrandSpec("Aurelio Audio", "luxury", ("Electronics",)),
    BrandSpec("Circuitry Co", "budget", ("Electronics",)),
    BrandSpec("Halcyon Tech", "mid", ("Electronics",)),
    # Apparel + Sports (deliberate overlap: cross-department brand affinity)
    BrandSpec("Stride", "mid", ("Apparel", "Sports")),
    BrandSpec("Peakline", "premium", ("Apparel", "Sports")),
    BrandSpec("Tempo Athletic", "mid", ("Apparel", "Sports")),
    BrandSpec("Basecamp", "premium", ("Sports", "Apparel")),
    BrandSpec("Everstep", "budget", ("Apparel",)),
    BrandSpec("Meridian Wear", "luxury", ("Apparel",)),
    BrandSpec("Cotton Union", "budget", ("Apparel",)),
    BrandSpec("Vertex Sport", "mid", ("Sports",)),
    BrandSpec("Trailhead", "budget", ("Sports",)),
    # Home + Beauty
    BrandSpec("Hearth & Oak", "premium", ("Home",)),
    BrandSpec("Copperpot", "mid", ("Home",)),
    BrandSpec("Linenfold", "luxury", ("Home",)),
    BrandSpec("Everyday Home", "budget", ("Home",)),
    BrandSpec("Lumen Skin", "premium", ("Beauty",)),
    BrandSpec("Botanic Field", "mid", ("Beauty",)),
    BrandSpec("Pure Ritual", "luxury", ("Beauty",)),
    BrandSpec("Daily Glow", "budget", ("Beauty",)),
    # Books, Toys, Pet
    BrandSpec("Northgate Press", "mid", ("Books",)),
    BrandSpec("Quill & Co", "premium", ("Books",)),
    BrandSpec("Paperlane", "budget", ("Books",)),
    BrandSpec("Brightblock", "mid", ("Toys",)),
    BrandSpec("Wondercraft", "premium", ("Toys",)),
    BrandSpec("Playbox", "budget", ("Toys",)),
    BrandSpec("Rufus & Co", "premium", ("Pet",)),
    BrandSpec("Whisker Works", "mid", ("Pet",)),
    BrandSpec("Petbasics", "budget", ("Pet",)),
)


#: Price multiplier and rating shift by brand tier.
TIER_PRICE_MULTIPLIER: dict[str, float] = {
    "budget": 0.55,
    "mid": 1.0,
    "premium": 1.75,
    "luxury": 3.10,
}

#: Mean product rating by tier, before per-product noise. Higher tiers rate
#: slightly better, which gives the quality signal something to correlate with.
TIER_RATING_MEAN: dict[str, float] = {
    "budget": 3.7,
    "mid": 4.0,
    "premium": 4.3,
    "luxury": 4.4,
}


#: Subcategory pairs bought together, with the probability that buying the
#: first pulls the second into the same order. This is the ground truth for
#: "frequently bought together" (FR-03): the diagnostic in `diagnostics.py`
#: checks that the co-purchase lift for these pairs is materially above the
#: lift for random pairs, which is what proves the signal is recoverable.
COMPLEMENTARY_PAIRS: tuple[tuple[str, str, float], ...] = (
    ("Running Shoes", "Athletic Socks", 0.42),
    ("Trail Shoes", "Athletic Socks", 0.34),
    ("Running Shoes", "Running Shorts", 0.26),
    ("Laptops", "Laptop Bags", 0.38),
    ("Laptops", "Mice", 0.30),
    ("Laptops", "Monitors", 0.18),
    ("Monitors", "Keyboards", 0.24),
    ("Smartphones", "Phone Cases", 0.55),
    ("Smartphones", "Screen Protectors", 0.44),
    ("Smartphones", "Chargers", 0.31),
    ("Headphones", "Audio Cables", 0.22),
    ("Speakers", "Audio Cables", 0.28),
    ("Coffee Makers", "Coffee Beans", 0.48),
    ("Sheet Sets", "Pillows", 0.33),
    ("Duvets", "Sheet Sets", 0.29),
    ("Cookware", "Knives", 0.21),
    ("Shampoo", "Conditioner", 0.62),
    ("Cleansers", "Moisturisers", 0.46),
    ("Moisturisers", "Serums", 0.33),
    ("Tents", "Sleeping Bags", 0.40),
    ("Backpacks", "Water Bottles", 0.27),
    ("Bike Helmets", "Bike Lights", 0.30),
    ("Cycling Jerseys", "Bike Lights", 0.15),
    ("Yoga Mats", "Resistance Bands", 0.25),
    ("Yoga Mats", "Foam Rollers", 0.22),
    ("Dog Food", "Dog Toys", 0.24),
    ("Cat Food", "Cat Litter", 0.51),
    ("Board Games", "Card Games", 0.19),
    ("Building Blocks", "Model Kits", 0.17),
    ("Winter Coats", "Fleece", 0.23),
)


#: Attribute value pools. Used both for structured attributes and for the
#: generated product description, so text embeddings and categorical features
#: describe the same product rather than two unrelated fictions.
ATTRIBUTE_VALUES: dict[str, tuple[str, ...]] = {
    "colour": ("black", "white", "navy", "grey", "olive", "burgundy", "sand", "teal"),
    "size": ("XS", "S", "M", "L", "XL", "XXL"),
    "material": ("cotton", "merino wool", "recycled polyester", "leather", "nylon", "bamboo"),
    "capacity": ("64GB", "128GB", "256GB", "512GB", "1TB"),
    "connectivity": ("Bluetooth 5.3", "USB-C", "Wi-Fi 6", "wired 3.5mm", "wireless 2.4GHz"),
    "skin_type": ("dry", "oily", "combination", "sensitive", "normal"),
    "hair_type": ("fine", "curly", "coloured", "thick", "damaged"),
    "volume": ("50ml", "100ml", "200ml", "400ml"),
    "spf": ("SPF 15", "SPF 30", "SPF 50"),
    "origin": ("Ethiopia", "Colombia", "Guatemala", "Kenya", "Brazil"),
    "roast": ("light roast", "medium roast", "dark roast"),
    "format": ("paperback", "hardcover", "audiobook"),
    "language": ("English", "Spanish", "German"),
    "players": ("2 players", "2-4 players", "3-6 players", "party size"),
    "age_range": ("ages 3+", "ages 6+", "ages 8+", "ages 12+", "adult"),
    "pieces": ("120 pieces", "500 pieces", "1000 pieces", "2000 pieces"),
    "weight": ("1kg", "3kg", "7kg", "12kg"),
    "life_stage": ("puppy", "kitten", "adult", "senior"),
}


def iter_subcategories() -> list[tuple[str, str, SubcategorySpec]]:
    """Flatten the taxonomy to (department, category, subcategory) triples."""
    rows: list[tuple[str, str, SubcategorySpec]] = []
    for department in DEPARTMENTS:
        for category in department.categories:
            for sub in category.subcategories:
                rows.append((department.name, category.name, sub))
    return rows


def brands_for_department(department: str) -> list[BrandSpec]:
    return [b for b in BRANDS if department in b.departments]


__all__ = [
    "ATTRIBUTE_VALUES",
    "BRANDS",
    "COMPLEMENTARY_PAIRS",
    "DEPARTMENTS",
    "TIER_PRICE_MULTIPLIER",
    "TIER_RATING_MEAN",
    "BrandSpec",
    "CategorySpec",
    "DepartmentSpec",
    "SubcategorySpec",
    "brands_for_department",
    "iter_subcategories",
]
