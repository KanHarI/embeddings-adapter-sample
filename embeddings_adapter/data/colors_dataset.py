import dataclasses


@dataclasses.dataclass
class Category:
    category_name: str
    category_items: list[str]


Dataset = list[Category]

RED = Category(
    category_name="Red",
    category_items=[
        "Apple",
        "Fire truck",
        "Tomato",
        "Stop sign",
        "Rose",
        "Strawberry",
        "Heart",
        "Chili pepper",
        "Cardinal bird",
        "Lipstick",
    ],
)
BLUE = Category(
    category_name="Blue",
    category_items=[
        "Sky",
        "Ocean",
        "Blueberry",
        "Jeans",
        "Hydrangea",
        "Whale",
        "Bluebird",
        "Sapphire",
        "Robin's egg",
        "Intel's logo",
    ],
)
YELLOW = Category(
    category_name="Yellow",
    category_items=[
        "Sun",
        "Banana",
        "Lemon",
        "School bus",
        "Sunflower",
        "Rubber duck",
        "Canary",
        "Corn",
        "Daffodil",
        "Taxi cab",
    ],
)
GREEN = Category(
    category_name="Green",
    category_items=[
        "Grass",
        "Trees",
        "Lime",
        "Broccoli",
        "Frog",
        "Dollar bills",
        "Emerald",
        "Shamrock",
        "Cucumber",
        "Avocado",
    ],
)
ORAGNE = Category(
    category_name="Orange",
    category_items=[
        "Orange (fruit)",
        "Pumpkin",
        "Carrot",
        "Tiger",
        "Traffic cone",
        "Goldfish",
        "Basketball",
        "Marigold",
        "Clownfish",
        "Monarch butterfly",
    ],
)
PURPLE = Category(
    category_name="Purple",
    category_items=[
        "Grape",
        "Lavender",
        "Plum",
        "Eggplant",
        "Orchid",
        "Amethyst",
        "Jelly",
        "Red Cabbage",
        "Purple finch",
        "Violet flower",
    ],
)
BROWN = Category(
    category_name="Brown",
    category_items=[
        "Chocolate",
        "Bear",
        "Tree trunk",
        "Leather jacket",
        "Coffee",
        "Acorn",
        "Potato",
        "Owl",
        "Walnut",
        "Deer",
    ],
)
BLACK = Category(
    category_name="Black",
    category_items=[
        "Coal",
        "Crow",
        "Tire",
        "Panther",
        "Short piano keys",
        "Hockey puck",
        "Night sky",
        "Black cat",
        "Batman",
    ],
)
WHITE = Category(
    category_name="White",
    category_items=[
        "Snow",
        "Cloud",
        "Swan",
        "Rice",
        "Tooth",
        "Wedding dress",
        "Sugar",
        "Salt",
        "Cotton",
        "Dove",
    ],
)
GRAY = Category(
    category_name="Gray",
    category_items=[
        "Elephant",
        "Mouse",
        "Concrete",
        "Pebble",
        "Shark",
        "Gray goo",
        "Battleship",
        "Wolf",
        "Koala",
        "Rhino",
    ],
)
COLORS_DATASET = Dataset(
    [
        RED,
        BLUE,
        YELLOW,
        GREEN,
        ORAGNE,
        PURPLE,
        BROWN,
        BLACK,
        WHITE,
        GRAY,
    ]
)
