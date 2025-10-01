import json
import os
import pickle


if __name__ == "__main__":
    with open(os.path.join("data", "output", "color-scheme.json"), "r") as input_handle:
        color_dict = json.load(input_handle)

    theme = dict()
    theme["light"] = dict()
    theme["dark"] = dict()

    theme["light"]["positive"] = color_dict["primary_green"][0]
    theme["light"]["negative"] = color_dict["primary_red"][0]
    theme["light"]["neutral"] = color_dict["effect"][0]
    theme["light"]["foreground"] = color_dict["black_light"][0]
    theme["light"]["background"] = color_dict["white"][0]
    theme["light"]["grey"] = color_dict["grey"][0]
    theme["light"]["cursor"] = color_dict["black"][0]

    theme["light"]["red"] = color_dict["secondary_red"][0]
    theme["light"]["green"] = color_dict["secondary_green"][0]
    theme["light"]["blue"] = color_dict["secondary_blue"][0]
    theme["light"]["yellow"] = color_dict["secondary_yellow"][0]
    theme["light"]["magenta"] = color_dict["secondary_magenta"][0]
    theme["light"]["cyan"] = color_dict["secondary_cyan"][0]

    theme["light"]["pastel_red"] = color_dict["secondary_red_muted"][0]
    theme["light"]["pastel_green"] = color_dict["secondary_green_muted"][0]
    theme["light"]["pastel_blue"] = color_dict["secondary_blue_muted"][0]
    theme["light"]["pastel_yellow"] = color_dict["secondary_yellow_muted"][0]
    theme["light"]["pastel_magenta"] = color_dict["secondary_magenta_muted"][0]
    theme["light"]["pastel_cyan"] = color_dict["secondary_cyan_muted"][0]

    theme["light"]["light_muted"] = color_dict["white_dark"][0]
    theme["light"]["dark_muted"] = color_dict["black_light"][0]
    theme["light"]["light_grey"] = color_dict["grey_light"][0]
    theme["light"]["dark_grey"] = color_dict["grey_dark"][0]

    theme["light"]["effect_bright"] = color_dict["effect_bright"][0]
    theme["light"]["effect_dark"] = color_dict["effect_dark"][0]
    theme["light"]["effect_muted"] = color_dict["secondary_effect"][0]
    theme["light"]["effect_pastel"] = color_dict["secondary_effect_muted"][0]

    theme["dark"]["positive"] = color_dict["primary_green"][0]
    theme["dark"]["negative"] = color_dict["primary_red"][0]
    theme["dark"]["neutral"] = color_dict["effect"][0]
    theme["dark"]["foreground"] = color_dict["white_dark"][0]
    theme["dark"]["background"] = color_dict["black"][0]
    theme["dark"]["grey"] = color_dict["grey"][0]
    theme["dark"]["cursor"] = color_dict["white"][0]

    theme["dark"]["red"] = color_dict["secondary_red_muted"][0]
    theme["dark"]["green"] = color_dict["secondary_green_muted"][0]
    theme["dark"]["blue"] = color_dict["secondary_blue_muted"][0]
    theme["dark"]["yellow"] = color_dict["secondary_yellow_muted"][0]
    theme["dark"]["magenta"] = color_dict["secondary_magenta_muted"][0]
    theme["dark"]["cyan"] = color_dict["secondary_cyan_muted"][0]

    theme["dark"]["pastel_red"] = color_dict["secondary_red"][0]
    theme["dark"]["pastel_green"] = color_dict["secondary_green"][0]
    theme["dark"]["pastel_blue"] = color_dict["secondary_blue"][0]
    theme["dark"]["pastel_yellow"] = color_dict["secondary_yellow"][0]
    theme["dark"]["pastel_magenta"] = color_dict["secondary_magenta"][0]
    theme["dark"]["pastel_cyan"] = color_dict["secondary_cyan"][0]

    theme["dark"]["light_muted"] = color_dict["black_light"][0]
    theme["dark"]["dark_muted"] = color_dict["white_dark"][0]
    theme["dark"]["light_grey"] = color_dict["grey_dark"][0]
    theme["dark"]["dark_grey"] = color_dict["grey_light"][0]

    theme["dark"]["effect_bright"] = color_dict["effect_dark"][0]
    theme["dark"]["effect_dark"] = color_dict["effect_bright"][0]
    theme["dark"]["effect_muted"] = color_dict["secondary_effect_muted"][0]
    theme["dark"]["effect_pastel"] = color_dict["secondary_effect"][0]

    with open(os.path.join("data", "output", "theme.pkl"), "wb") as output_handle:
        pickle.dump(theme, output_handle)
