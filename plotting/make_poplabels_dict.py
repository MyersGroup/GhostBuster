def make_poplabels_dict():
    poplabels_dict = {}
    p1 = [
        "esn",
        "gwd",
        "msl",
        "lwk",
        "mbutipygmy",
        "biakapygmy",
        "san",
        "mandenka",
        "bantusafrica",
        "bantukenya",
        "yoruba",
        "I10871",
        "I5950",
        "ela001",
        "new001",
        "baa001",
        "Somali",
        "Luo",
        "Masai",
        "Ju_hoan_North",
        "Dinka",
        "Khomani_San",
    ]

    p2 = [
        "ESN",
        "GWD",
        "MSL",
        "LWK",
        "Mbuti",
        "Biaka",
        "San",
        "Mandenka",
        "Bantu SAfrica",
        "Bantu Kenya",
        "Yoruba",
        "Shum Laka (7890 y.a.)",
        "Mota (4472 y.a.)",
        "Eland (493 y.a.) ",
        "Newcastle (418 y.a.)",
        "Ballito Bay (1909 y.a.)",
        "Somali",
        "Luo",
        "Masai",
        "Ju hoan North",
        "Dinka",
        "Khomani San",
    ]

    for i, j in zip(p1, p2):
        poplabels_dict[i] = j

    return poplabels_dict
