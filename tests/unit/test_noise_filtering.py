from fanatic.preprocess import labels, filter_data

INPUT_DATA = {
    "bayarea": [
        {"text": "Place where my age group goes.", "id": "eusrb"},
        {"text": "The San Francisco Tape Music Festival looks cool!!", "id": "euomq"},
        {
            "text": "My girlfriend and I, bayarea residents, have nothing to do and no good ideas! help us out!",
            "id": "eume4",
        },
    ],
    "tattoos": [
        {
            "text": 'I\'m getting the "Reddit coat of arms" tattoo in a couple hours',
            "id": "eur8o",
        }
    ],
    "personalfinance": [
        {"text": "Selling Equity Stock vs. Options", "id": "euq0e"},
        {
            "text": "Great site to research credit building/repair yourself",
            "id": "euhnc",
        },
    ],
    "devils": [{"text": "Thought you might enjoy this...", "id": "eun3x"}],
    "Screenwriting": [
        {"text": "Which program do you use for screenwriting?", "id": "eulyn"},
        {
            "text": 'Chris "Buried" Sparling\'s letter to the Academy backfires ',
            "id": "eujze",
        },
    ],
}

OUTPUT_DATA = {
    "tattoos": [
        {
            "text": 'I\'m getting the "Reddit coat of arms" tattoo in a couple hours',
            "id": "eur8o",
        }
    ],
    "personalfinance": [
        {"text": "Selling Equity Stock vs. Options", "id": "euq0e"},
        {
            "text": "Great site to research credit building/repair yourself",
            "id": "euhnc",
        },
    ],
}


def test_noise_filtering():
    # GIVEN
    subreddit_labels_file = "subreddit_labels.json"  # in FANATIC base dir
    subreddit_noise_percentage = 0.5
    seed = 42

    # WHEN
    subreddit_labels = labels.load_subreddit_labels(subreddit_labels_file)
    data = filter_data.filter_data_by_noise_percentage(
        INPUT_DATA,
        n_read=3,
        subreddit_noise_percentage=subreddit_noise_percentage,
        subreddit_labels=subreddit_labels,
        seed=seed,
    )

    # THEN
    assert data == OUTPUT_DATA
