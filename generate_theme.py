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

    theme["dark"]["positive"] = color_dict["primary_green"][0]
    theme["dark"]["negative"] = color_dict["primary_red"][0]
    theme["dark"]["neutral"] = color_dict["effect"][0]
    theme["dark"]["foreground"] = color_dict["white_dark"][0]
    theme["dark"]["background"] = color_dict["black"][0]
    theme["dark"]["grey"] = color_dict["grey"][0]
    theme["dark"]["cursor"] = color_dict["white"][0]

    with open(os.path.join("data", "output", "theme.pkl"), "wb") as output_handle:
        pickle.dump(theme, output_handle)
