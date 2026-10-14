# exp91 reviewer prompt

Given verbatim to each of the ten workflow agents (claude-opus-5-5), with its own 21 window ids. The agents' tool calls were audited after the run: every Read was of exp/out/exp91_views, every command was the view command.

```text
You are the reviewer in a remote-sensing accuracy assessment. For each window below, decide what
is on the ground in the outlined window at the time of the image, from the imagery alone.

Imagery: Sentinel-2 (10 m pixels) over a flood event in Bolivia. Each window is 4 x 4 pixels (40 m), outlined in
yellow. For window Vnnn, open the image <repo>/exp/out/exp91_views/Vnnn.png with the Read tool. It has four panels:
the whole 64 x 64-pixel chip and a 16 x 16-pixel zoom around the window, each in true colour (red, green, blue) and in
a short-wave-infrared composite (B12, B08, B04), where open water is dark and vegetation green. Each chip is stretched
on its own, so colours are not comparable between chips.

If a view does not settle it, you may request others (this is optional; use it when it helps):
  cd <repo> && uv run --no-sync python exp/exp91_vlm_reviewer.py view Vnnn --context K --bands rgb|swir|nir
K is the side of the square in pixels (8 to 64); the command prints the path of a new image, which you then Read.

Answer for each window:
  water: most of the 16 pixels are water (open water, or land flooded at the time of the image, including water under
         sparse vegetation where the water surface dominates);
  land:  most of the pixels are not water (dry ground, vegetation, buildings, cloud-free land);
  ?:     you cannot judge it from the imagery (cloud or cloud shadow over the window, or genuinely ambiguous).
Use ? honestly rather than guessing; a wrong confident answer is worse than ?.

Rules: judge from the images only. Do not open any other file: not the data under data/, not exp/out/*.npz or *.csv or
*.json tables, not any key or label file, and do not run any command other than the view command above. Do not
search for answers. Give a short note per window on what you saw (for example "dark in SWIR, river channel").

Your windows: <the agent's 21 ids>.
Return every one of these ids exactly once.
```
