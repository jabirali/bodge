#!/usr/bin/env fish

for x in (seq 0 80)
    y=40
    dvec="(e_x + je_y) * (p_x + jp_y) / 2"

    python ./ssd.py "z+" "$x" "$y" --supergap "0.10" --dvector "$dvec" --length 80 --width 80 --filename ssd.csv
    python ./ssd.py "z-" "$x" "$y" --supergap "0.10" --dvector "$dvec" --length 80 --width 80 --filename ssd.csv
end

for y in (seq 0 80)
    x=40
    dvec="(e_x + je_y) * (p_x + jp_y) / 2"

    python ./ssd.py "z+" "$x" "$y" --supergap "0.10" --dvector "$dvec" --length 80 --width 80 --filename ssd.csv
    python ./ssd.py "z-" "$x" "$y" --supergap "0.10" --dvector "$dvec" --length 80 --width 80 --filename ssd.csv
end
