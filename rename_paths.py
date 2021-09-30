# Per rinominare i path da usare su wls

import argparse
import json
import platform
from init import write_json


def rename_paths(file, heading):
    with open(file) as json_file:
        d = json.load(json_file)

    json_file.close()

    if platform.system() == 'Windows' and '\\' in heading:
        heading = heading.replace('\\', '/')
        print("heading traformato: ", heading)

    if platform.system() == 'Windows' and '\\' in file:
        file = file.replace('\\', '/')

    spl = file.split('/')
    stat = spl[len(spl) - 1]
    print("STAT: ", stat)

    out_file = file[:file.index(stat)] + 'dict_path_wls.json'
    print("OUT: ", out_file)

    for i in d.keys():
        p = d[i]  # path attualmente sul file
        print("p: ", p)

        if type(p) is str:
            if heading in p:
                print("no changes")
            else:
                spl = p.split('/')
                # tutto ciò che è prima di stat (stat escluso) lo cambio, quello che c'è dopo lo tengo
                # stat =
                if len(spl) > 1:
                    stat = spl[1]
                    print("STAT: ", stat)

                    p = p.replace(p[:p.index(stat)], heading)  # index restituisce l'indice a cui stat inzia
                    # p = heading + stat + '/' + spl[len(spl) - 1]
                    print("p trasformato: ", p)
                    print()

        d[i] = p

    write_json(d, out_file)  # salvo il file con i path modificati


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rename paths")

    parser.add_argument("--json_file", dest="input", default=None, help="Path of the json file with the paths to rename")
    parser.add_argument("--heading", dest="heading", default=0, help="Heading to replace")

    args = parser.parse_args()

    rename_paths(file=args.input, heading=args.heading)