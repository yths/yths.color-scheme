import os

import colour
import loguru
import numpy

def main(output_file_path: str = os.path.join("data", "output", "colors.npy")):
    if not os.path.exists(output_file_path):
        os.makedirs(os.path.dirname(output_file_path))
        colors_sRGB = []
        for r in range(256):
            for g in range(256):
                for b in range(256):
                    color_sRGB = [r / 255.0, g / 255.0, b / 255.0]
                    colors_sRGB.append(color_sRGB)

        colors_sRGB = numpy.array(colors_sRGB)
        colors_LCHab = colour.convert(colors_sRGB, "sRGB", "CIE LCHab")
        
        loguru.logger.info(f"Conversion complete, total colors: {len(colors_LCHab)}")

        colors = numpy.hstack((colors_sRGB, colors_LCHab))
        numpy.save(output_file_path, colors)
        loguru.logger.info(f"Colors saved to {output_file_path}")
    else:
        loguru.logger.info(f"Colors already exist at {output_file_path}, skipping generation.")

if __name__ == "__main__":
    main()

    colors = numpy.load(os.path.join("data", "output", "colors.npy"))
    # min, max, mean, median for each channel
    for i, channel in enumerate(["R", "G", "B", "L", "C", "H"]):
        loguru.logger.info(
            f"{channel}: min={colors[:, i].min()}, max={colors[:, i].max()}, mean={colors[:, i].mean()}, median={numpy.median(colors[:, i])}"
        )

    # plot histogram for H channel, 360 bins

    import matplotlib.pyplot as plt
    plt.hist(colors[:, 5], bins=3600, range=(0, 1))
    plt.title("Hue Channel Histogram")
    plt.xlabel("Hue (degrees)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join("data", "output", "hue_histogram.png"))
    plt.show()

    # divide the colors array into 3600 bins based on the H channel
    bins = [[] for _ in range(3600)]
    for color in colors:
        h = color[5]
        bin_index = int(h * 3600) % 3600
        bins[bin_index].append(color)

    bins = [numpy.array(sorted(bin, key=lambda c: c[3] + c[4])) for bin in bins]

    # iterate over each bin, find the color with the highest combined L and C value and check if a color with the same L and C value exists in a bin 1200 bins away in each direction, allow for a small deviation epsilon in L and C values, return all triples that meet this crieria
    epsilon = 75e-5
    triples = []
    for i in range(3600):
        if len(bins[i]) == 0:
            continue
        bin_colors = numpy.array(bins[i])
        lc_values = bin_colors[:, 3] + bin_colors[:, 4]
        max_index = numpy.argmax(lc_values)
        max_color = bin_colors[max_index]
        L, c, h = max_color[3], max_color[4], max_color[5]
        triple = [max_color]
        for offset in [-1200, 1200]:
            neighbor_index = (i + offset) % 3600
            if len(bins[neighbor_index]) == 0:
                continue
            neighbor_colors = bins[neighbor_index]
            for neighbor_color in neighbor_colors:
                if abs(neighbor_color[3] - L) < epsilon and abs(neighbor_color[4] - c) < epsilon:
                    triple.append(neighbor_color)
                    break
        if len(triple) == 3:
            triples.append(triple)
    
    # return the triple with the highest combined L and C value
    if len(triples) == 0:
        loguru.logger.info("No triples found")
    else:
        best_triple = max(triples, key=lambda t: t[0][3] + t[0][4])
        loguru.logger.info(f"Best triple found with L={best_triple[0][3]}, C={best_triple[0][4]}, H={best_triple[0][5]}")
        for color in best_triple:
            loguru.logger.info(f"Color: R={color[0]*255}, G={color[1]*255}, B={color[2]*255}, L={color[3]}, C={color[4]}, H={color[5]*360}")

        # find the minimum hue of the best triple
        hues = [color[5] for color in best_triple]
        min_hue = min(hues)
        loguru.logger.info(f"Minimum hue of best triple: {min_hue*360}")
        
        # find the closes grey to the average of the best triple
        avg_L = numpy.mean([color[3] for color in best_triple])
        avg_C = numpy.mean([color[4] for color in best_triple])
        grey = numpy.array([0.5, 0.5, 0.5])
        grey_LCHab = colour.convert(grey, "sRGB", "CIE LCHab")
        grey_LCHab[0] = avg_L
        grey_LCHab[1] = 0.0
        grey_sRGB = colour.convert(grey_LCHab, "CIE LCHab", "sRGB")
        loguru.logger.info(f"Closest grey to average of best triple: R={grey_sRGB[0]*255}, G={grey_sRGB[1]*255}, B={grey_sRGB[2]*255}, L={grey_LCHab[0]}, C={grey_LCHab[1]}, H={grey_LCHab[2]*360}")


        # determine the index of the triplet has the smallest max(r, g, b) value - this color has the most potential to become brighter
        max_rgb = [color[0] + color[1] + color[2] for color in best_triple]
        min_max_rgb_index = numpy.argmin(max_rgb)
        loguru.logger.info(f"Index of color in best triple with smallest max(R, G, B): {min_max_rgb_index}, max(R, G, B)={max_rgb[min_max_rgb_index]}")

        most_potential = best_triple[min_max_rgb_index]
        C = most_potential[4]
        H = most_potential[5]

        # find the brightest primary
        L = 1
        brightest_primary = colour.convert([L, C, H], "CIE LCHab", "sRGB")
        r, g, b = brightest_primary
        while r > 1.0 or g > 1.0 or b > 1.0:
            L *= 0.999
            brightest_primary = colour.convert([L, C, H], "CIE LCHab", "sRGB")
            r, g, b = brightest_primary
        brightest_primary_LCHab = colour.convert(brightest_primary, "sRGB", "CIE LCHab") 
        loguru.logger.info(f"Max L for C={C}, H={H} is {L}, RGB=({r*255}, {g*255}, {b*255})")

        # find the closes white to the grey
        white = numpy.array([1.0, 1.0, 1.0])
        white_LCHab = colour.convert(white, "sRGB", "CIE LCHab")
        white_LCHab[0] = brightest_primary_LCHab[0]
        white_sRGB = colour.convert(white_LCHab, "CIE LCHab", "sRGB")
        r, g, b = 256, 256, 256
        while r > 255 or g > 255 or b > 255 or r < 0 or g < 0 or b < 0:
            white_LCHab[0] *= 0.999
            white_LCHab[1] = 0.0
            white_LCHab[2] = grey_LCHab[2]
            white_sRGB = colour.convert(white_LCHab, "CIE LCHab", "sRGB")
            r, g, b = white_sRGB
        loguru.logger.info(f"Closest white to average of best triple: R={white_sRGB[0]*255}, G={white_sRGB[1]*255}, B={white_sRGB[2]*255}, L={white_LCHab[0]}, C={white_LCHab[1]}, H={white_LCHab[2]*360}")

        # find the black that is the inverse of the white
        black = numpy.array([0.0, 0.0, 0.0])
        black_LCHab = colour.convert(black, "sRGB", "CIE LCHab")
        black_sRGB = colour.convert(black_LCHab, "CIE LCHab", "sRGB")
        black_LCHab[0] = grey_LCHab[0] - (white_LCHab[0] - grey_LCHab[0])
        black_LCHab[1] = 0.0
        black_LCHab[2] = grey_LCHab[2]
        black_sRGB = colour.convert(black_LCHab, "CIE LCHab", "sRGB")
        r, g, b = -1, -1, -1
        while r < 0 or g < 0 or b < 0:
            black_LCHab[0] *= 1.001
            black_LCHab[1] = 0.0
            black_LCHab[2] = grey_LCHab[2]
            black_sRGB = colour.convert(black_LCHab, "CIE LCHab", "sRGB")
            r, g, b = black_sRGB
        loguru.logger.info(f"Closest black to average of best triple: R={black_sRGB[0]*255}, G={black_sRGB[1]*255}, B={black_sRGB[2]*255}, L={black_LCHab[0]}, C={black_LCHab[1]}, H={black_LCHab[2]*360}")


        # darkest primary color
        darkest_primary_LCHab = brightest_primary_LCHab.copy()
        darkest_primary_LCHab[0] = black_LCHab[0]
        darkest_primary = colour.convert(darkest_primary_LCHab, "CIE LCHab", "sRGB")
        loguru.logger.info(f"Darkest primary color: R={darkest_primary[0]*255}, G={darkest_primary[1]*255}, B={darkest_primary[2]*255}, L={darkest_primary_LCHab[0]}, C={darkest_primary_LCHab[1]}, H={darkest_primary_LCHab[2]*360}")


        # complementary color
        complementary_color_LCHab = numpy.array([grey_LCHab[0], darkest_primary_LCHab[1], abs(darkest_primary_LCHab[2] - 0.5)])

        brightest_complementary_color_LCHab = complementary_color_LCHab.copy()
        brightest_complementary_color_LCHab[0] = brightest_primary_LCHab[0]

        brigtest_complementary_color_LCHab_candidate1 = brightest_complementary_color_LCHab.copy()
        brigtest_complementary_color_LCHab_candidate2 = brightest_complementary_color_LCHab.copy()

        adjusted_candidates = []
        for a in [brigtest_complementary_color_LCHab_candidate1, brigtest_complementary_color_LCHab_candidate2]:
            brightest_complementary_color_sRGB = colour.convert(a, "CIE LCHab", "sRGB")
            rl, gl, bl = brightest_complementary_color_sRGB
            darkest_complementary_color_LCHab = a.copy()
            darkest_complementary_color_LCHab[0] = darkest_primary_LCHab[0]
            darkest_complementary_color_sRGB = colour.convert(darkest_complementary_color_LCHab, "CIE LCHab", "sRGB")
            rd, gd, bd = darkest_complementary_color_sRGB
            while rl > 1.0 or gl > 1.0 or bl > 1.0 or rd < 0.0 or gd < 0.0 or bd < 0.0:
                a[2] = a[2] + 0.001
                if a[2] > 1.0:
                    a[2] = 0.0
                brightest_complementary_color_sRGB = colour.convert(a, "CIE LCHab", "sRGB")
                rl, gl, bl = brightest_complementary_color_sRGB
                darkest_complementary_color_LCHab = a.copy()
                darkest_complementary_color_LCHab[0] = darkest_primary_LCHab[0]
                darkest_complementary_color_sRGB = colour.convert(darkest_complementary_color_LCHab, "CIE LCHab", "sRGB")
                rd, gd, bd = darkest_complementary_color_sRGB

            adjusted_candidates.append(a)
            
        brightest_complementary_color_LCHab = sorted(adjusted_candidates, key=lambda c: abs(c[2] - complementary_color_LCHab[2]))[0]
        brightest_complementary_color_sRGB = colour.convert(brightest_complementary_color_LCHab, "CIE LCHab", "sRGB")


        loguru.logger.info(f"Brightest complementary color: R={brightest_complementary_color_sRGB[0]*255}, G={brightest_complementary_color_sRGB[1]*255}, B={brightest_complementary_color_sRGB[2]*255}, L={brightest_complementary_color_LCHab[0]}, C={brightest_complementary_color_LCHab[1]}, H={brightest_complementary_color_LCHab[2]*360}")

        darkest_complementary_color_LCHab = brightest_complementary_color_LCHab.copy()
        darkest_complementary_color_LCHab[0] = 0.4227785339122564
        darkest_complementary_color_sRGB = colour.convert(darkest_complementary_color_LCHab, "CIE LCHab", "sRGB")
        loguru.logger.info(f"Darkest complementary color: R={darkest_complementary_color_sRGB[0]*255}, G={darkest_complementary_color_sRGB[1]*255}, B={darkest_complementary_color_sRGB[2]*255}, L={darkest_complementary_color_LCHab[0]}, C={darkest_complementary_color_LCHab[1]}, H={darkest_complementary_color_LCHab[2]*360}")
        
        complementary_color_LCHab[2] = darkest_complementary_color_LCHab[2]
        complementary_color_sRGB = colour.convert(complementary_color_LCHab, "CIE LCHab", "sRGB")
        loguru.logger.info(f"Complementary color: R={complementary_color_sRGB[0]*255}, G={complementary_color_sRGB[1]*255}, B={complementary_color_sRGB[2]*255}, L={complementary_color_LCHab[0]}, C={complementary_color_LCHab[1]}, H={complementary_color_LCHab[2]*360}")
    

        
        C = complementary_color_LCHab[1]
        hues = list(range(0, 360, 120))
        secondary_colors = []
        hues_labels = []
        while len(secondary_colors) < 3 and C > 0:
            secondary_colors = []
            hues_labels = []
            for h in hues:
                c_LCh = numpy.array([complementary_color_LCHab[0], C, ((h + 60 + min_hue) % 360) / 360.0])
                c_sRGB = colour.convert(c_LCh, "CIE LCHab", "sRGB")
                r, g, b = c_sRGB
                if r > 1.0 or g > 1.0 or b > 1.0 or r < 0.0 or g < 0.0 or b < 0.0:
                    continue
                secondary_colors.append(c_sRGB)
                hues_labels.append(c_LCh)
            C -= 0.1

        muted_C = None
        for color in secondary_colors:
            r, g, b = color
            L, C, H = colour.convert(color, "sRGB", "CIE LCHab")
            muted_C = C
            loguru.logger.info(f"Color found: R={r*255}, G={g*255}, B={b*255}, L={L}, C={C}, H={H*360}")

        if muted_C > best_triple[0][4] * (2 / 3):
            muted_C = best_triple[0][4] * (2 / 3)

        new_secondary_colors = []
        new_hues_labels = []
        for color in secondary_colors:
            L, C, H = colour.convert(color, "sRGB", "CIE LCHab")
            color_rgb = colour.convert([L, muted_C, H], "CIE LCHab", "sRGB")
            new_secondary_colors.append(color_rgb)
            new_hues_labels.append([L, muted_C, H])
        secondary_colors = new_secondary_colors
        hues_labels = new_hues_labels

        best_triple_muted = []
        for color in best_triple:
            L, C, H = color[3], color[4], color[5]
            color_muted = colour.convert([L, muted_C, H], "CIE LCHab", "sRGB")
            best_triple_muted.append([L, muted_C, H] + list(color_muted))

        muted_complementary_color_LCHab = complementary_color_LCHab.copy()
        muted_complementary_color_LCHab[1] = muted_C
        muted_complementary_color_sRGB = colour.convert(muted_complementary_color_LCHab, "CIE LCHab", "sRGB")
        loguru.logger.info(f"Muted complementary color: R={muted_complementary_color_sRGB[0]*255}, G={muted_complementary_color_sRGB[1]*255}, B={muted_complementary_color_sRGB[2]*255}, L={muted_complementary_color_LCHab[0]}, C={muted_complementary_color_LCHab[1]}, H={muted_complementary_color_LCHab[2]*360}")


        # create even more muted version of best triple and secondary colors
        even_more_muted_C = muted_C * 0.5
        more_muted_colors = []
        for color in best_triple_muted:
            L, C, H = color[0], color[1], color[2]
            color_even_more_muted = colour.convert([L, even_more_muted_C, H], "CIE LCHab", "sRGB")
            more_muted_colors.append([L, even_more_muted_C, H] + list(color_even_more_muted))
            loguru.logger.info(f"Even more muted color: R={color_even_more_muted[0]*255}, G={color_even_more_muted[1]*255}, B={color_even_more_muted[2]*255}, L={L}, C={even_more_muted_C}, H={H*360}")
        for color in secondary_colors:
            L, C, H = colour.convert(color, "sRGB", "CIE LCHab")
            color_even_more_muted = colour.convert([L, even_more_muted_C, H], "CIE LCHab", "sRGB")
            more_muted_colors.append([L, even_more_muted_C, H] + list(color_even_more_muted))
            loguru.logger.info(f"Even more muted secondary color: R={color_even_more_muted[0]*255}, G={color_even_more_muted[1]*255}, B={color_even_more_muted[2]*255}, L={L}, C={even_more_muted_C}, H={H*360}")

        even_more_muted_complementary_color_LCHab = complementary_color_LCHab.copy()
        even_more_muted_complementary_color_LCHab[1] = even_more_muted_C
        even_more_muted_complementary_color_sRGB = colour.convert(even_more_muted_complementary_color_LCHab, "CIE LCHab", "sRGB")
        loguru.logger.info(f"Even more muted complementary color: R={even_more_muted_complementary_color_sRGB[0]*255}, G={even_more_muted_complementary_color_sRGB[1]*255}, B={even_more_muted_complementary_color_sRGB[2]*255}, L={even_more_muted_complementary_color_LCHab[0]}, C={even_more_muted_complementary_color_LCHab[1]}, H={even_more_muted_complementary_color_LCHab[2]*360}")

        more_muted_colors.append([even_more_muted_complementary_color_LCHab[0], even_more_muted_C, even_more_muted_complementary_color_LCHab[2]] + list(even_more_muted_complementary_color_sRGB))

        # create grey levels
        L_levels = numpy.linspace(black_LCHab[0], white_LCHab[0], num=7)

        light_black = black_LCHab.copy()
        light_black[0] = L_levels[1]
        light_black_sRGB = colour.convert(light_black, "CIE LCHab", "sRGB")
        loguru.logger.info(f"Lighter black: R={light_black_sRGB[0]*255}, G={light_black_sRGB[1]*255}, B={light_black_sRGB[2]*255}, L={light_black[0]}, C={light_black[1]}, H={light_black[2]*360}")

        lighter_black = black_LCHab.copy()
        lighter_black[0] = L_levels[2]
        lighter_black_sRGB = colour.convert(lighter_black, "CIE LCHab", "sRGB")
        loguru.logger.info(f"Lighter black: R={lighter_black_sRGB[0]*255}, G={lighter_black_sRGB[1]*255}, B={lighter_black_sRGB[2]*255}, L={lighter_black[0]}, C={lighter_black[1]}, H={lighter_black[2]*360}")
    
        dark_white = white_LCHab.copy()
        dark_white[0] = L_levels[5]
        dark_white_sRGB = colour.convert(dark_white, "CIE LCHab", "sRGB")
        loguru.logger.info(f"Dark white: R={dark_white_sRGB[0]*255}, G={dark_white_sRGB[1]*255}, B={dark_white_sRGB[2]*255}, L={dark_white[0]}, C={dark_white[1]}, H={dark_white[2]*360}")

        darker_white = white_LCHab.copy()
        darker_white[0] = L_levels[4]
        darker_white_sRGB = colour.convert(darker_white, "CIE LCHab", "sRGB")
        loguru.logger.info(f"Darker white: R={darker_white_sRGB[0]*255}, G={darker_white_sRGB[1]*255}, B={darker_white_sRGB[2]*255}, L={darker_white[0]}, C={darker_white[1]}, H={darker_white[2]*360}")

        # visualize the best triple and grey and white and black and brightest and darkest primary
        # show the secondary colors in a second row
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(5, 8, figsize=(24, 12))
        ax[0, 3].axis("off")
        ax[0, 4].axis("off")
        ax[0, 5].axis("off")
        ax[0, 7].axis("off")
        ax[1, 7].axis("off")
        ax[2, 7].axis("off")
        ax[3, 7].axis("off")
        ax[4, 1].axis("off")
        ax[4, 3].axis("off")
        ax[4, 4].axis("off")
        ax[4, 6].axis("off")


        for i, color in enumerate(best_triple):
            ax[0, i].imshow([[color[:3]]])
            ax[0, i].set_title(f"L={color[3]:.2f}\nC={color[4]:.2f}\nH={color[5]*360:.1f}")
            ax[0, i].axis("off")
        for i, color in enumerate(best_triple_muted):
            ax[1, i].imshow([[color[3:]]])
            ax[1, i].set_title(f"Muted\nL={color[0]:.2f}\nC={color[1]:.2f}\nH={color[2]*360:.1f}")
            ax[1, i].axis("off")
        for i, color in enumerate(more_muted_colors):
            ax[2, i].imshow([[color[3:]]])
            ax[2, i].set_title(f"More Muted\nL={color[0]:.2f}\nC={color[1]:.2f}\nH={color[2]*360:.1f}")
            ax[2, i].axis("off")
        ax[3, 3].imshow([[grey_sRGB]])
        ax[3, 3].set_title(f"Grey\nL={grey_LCHab[0]:.2f}\nC={grey_LCHab[1]:.2f}\nH={grey_LCHab[2]*360:.1f}")
        ax[3, 3].axis("off")
        ax[3, 0].imshow([[white_sRGB]])
        ax[3, 0].set_title(f"White\nL={white_LCHab[0]:.2f}\nC={white_LCHab[1]:.2f}\nH={white_LCHab[2]*360:.1f}")
        ax[3, 0].axis("off")
        ax[3, 1].imshow([[dark_white_sRGB]])
        ax[3, 1].set_title(f"Dark White\nL={dark_white[0]:.2f}\nC={dark_white[1]:.2f}\nH={dark_white[2]*360:.1f}")
        ax[3, 1].axis("off")
        ax[3, 2].imshow([[darker_white_sRGB]])
        ax[3, 2].set_title(f"Darker White\nL={darker_white[0]:.2f}\nC={darker_white[1]:.2f}\nH={darker_white[2]*360:.1f}")
        ax[3, 2].axis("off")
        ax[3, 4].imshow([[lighter_black_sRGB]])
        ax[3, 4].set_title(f"Lighter Black\nL={light_black[0]:.2f}\nC={light_black[1]:.2f}\nH={light_black[2]*360:.1f}")
        ax[3, 4].axis("off")
        ax[3, 5].imshow([[light_black_sRGB]])
        ax[3, 5].set_title(f"Light Black\nL={light_black[0]:.2f}\nC={light_black[1]:.2f}\nH={light_black[2]*360:.1f}")
        ax[3, 5].axis("off")
        ax[3, 6].imshow([[black_sRGB]])
        ax[3, 6].set_title(f"Black\nL={black_LCHab[0]:.2f}\nC={black_LCHab[1]:.2f}\nH={black_LCHab[2]*360:.1f}")
        ax[3, 6].axis("off")
        ax[4, 0].imshow([[brightest_primary]])
        ax[4, 0].set_title(f"Brightest Primary\nL={brightest_primary_LCHab[0]:.2f}\nC={brightest_primary_LCHab[1]:.2f}\nH={brightest_primary_LCHab[2]*360:.1f}")
        ax[4, 0].axis("off")
        ax[4, 2].imshow([[darkest_primary]])
        ax[4, 2].set_title(f"Darkest Primary\nL={darkest_primary_LCHab[0]:.2f}\nC={darkest_primary_LCHab[1]:.2f}\nH={darkest_primary_LCHab[2]*360:.1f}")
        ax[4, 2].axis("off")
        ax[0, 6].imshow([[complementary_color_sRGB]])
        ax[0, 6].set_title(f"Complementary Color\nL={complementary_color_LCHab[0]:.2f}\nC={complementary_color_LCHab[1]:.2f}\nH={complementary_color_LCHab[2]*360:.1f}")
        ax[0, 6].axis("off")
        ax[4, 5].imshow([[brightest_complementary_color_sRGB]])
        ax[4, 5].set_title(f"Brightest Complementary\nL={brightest_complementary_color_LCHab[0]:.2f}\nC={brightest_complementary_color_LCHab[1]:.2f}\nH={brightest_complementary_color_LCHab[2]*360:.1f}")
        ax[4, 5].axis("off")
        ax[4, 7].imshow([[darkest_complementary_color_sRGB]])
        ax[4, 7].set_title(f"Darkest Complementary\nL={darkest_complementary_color_LCHab[0]:.2f}\nC={darkest_complementary_color_LCHab[1]:.2f}\nH={darkest_complementary_color_LCHab[2]*360:.1f}")
        ax[4, 7].axis("off")
        for i, color in enumerate(secondary_colors):
            ax[1, i + 3].imshow([[color]])
            ax[1, i + 3].set_title(f"Secondary Color\nL={hues_labels[i][0]:.2f}\nC={hues_labels[i][1]:.2f}\nH={hues_labels[i][2]*360:.1f}")
            ax[1, i + 3].axis("off")
        # muted complementary color
        ax[1, 6].imshow([[muted_complementary_color_sRGB]])
        ax[1, 6].set_title(f"Muted Complementary\nL={muted_complementary_color_LCHab[0]:.2f}\nC={muted_complementary_color_LCHab[1]:.2f}\nH={muted_complementary_color_LCHab[2]*360:.1f}")
        ax[1, 6].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join("data", "output", "best_triple.png"))
        plt.show() 


