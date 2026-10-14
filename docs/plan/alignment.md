# Does the map line up with its labels? A shift check on exp98 and exp99 (exp100 preregistration)

**Status: frozen on 10 October 2026, before any shifted reading of exp98's or exp99's maps was computed.** Written
10 October 2026, after the owner approved the check ("yes"); the half-pixel bar and the 3-pixel search are the
record's choice. The labels of exp98 and exp99 have been read: both experiments are recorded. This check reads the
same maps at the same points, moved by whole pixels, and asks only where the map and the labels agree best.

**Result, 10 October 2026 (added at recording; the frozen text is unchanged).** Run once at 70b2c01 (job 1272189) and
recorded after an independent audit ([comparisons](../results/comparisons.md#does-the-map-line-up-with-its-labels-a-shift-check-on-exp98-and-exp99-exp100)).
The classes at the labelled pixels gave the recorded errors in every set. T: not determined (flat within about a pixel;
intervals reach about 1.1 px). V: offset, 1.9 px east (90% interval 0.7 to 2.4), agreement 89.2% there against 87.3% at
the pixel, a position chosen on these points; exp98's verdicts stand, as this page says. S (STRICT and LENIENT): not
determined. Changes after
freezing are listed under Deviations.

- **Allowed before freezing:** the smoke on synthetic inputs (`python exp/exp100_alignment.py --smoke`,
  `tests/test_exp100.py`).
- **Not allowed before freezing:** any reading of exp98's or exp99's maps at a shifted position. The script refuses
  the run until this page says frozen.

## The question

The owner asked whether the satellite imagery and the label points are correctly matched in position: if the map and
the labels were offset from each other, every accuracy and ranking in exp98 and exp99 would be measured on the wrong
pixels. Two different things can be offset:

- **The record's code** that finds a label's pixel on the map. exp98's labels and map are in the same CRS (EPSG:32737)
  on the same 10 m lattice, and its run recorded that all 1,042 located points sit at the middle of a score pixel
  (largest departure 0.0 px), so a mistake there would have to be an off-by-one, not a half pixel. exp99 reprojects
  each TimeSync plot from WGS84 longitude and latitude onto the map's UTM grid.
- **The data:** Sentinel-2's own geolocation, and the imagery the labels were drawn on (Planet for AWF's experts;
  Landsat and Google Earth's high-resolution imagery for TimeSync's interpreters).

## The measure

For each point, the map is read at its own pixel and at every pixel moved by `(south, east)` whole pixels, each from
-3 to +3 (49 positions, up to about 42 m diagonally). At each position the score is the probability the map gives to
the classes the label accepts: the labelled class for AWF; under exp99's crosswalk, the AWF classes STRICT (primary)
or LENIENT accepts for the TimeSync class. The probability is used rather than agreement because it changes smoothly
between neighbouring pixels; agreement (the map's class accepted) is reported beside it.

- **The surface:** the mean score over the points at each of the 49 positions. Only points whose 7 x 7 block is
  covered in the grid that holds their own pixel enter, so every position averages the same points; the others are
  counted.
- **The offset estimate:** the position of the surface's maximum, refined on each axis by a parabola through the
  maximum and its two neighbours (the standard sub-pixel peak; no refinement where the maximum is on the edge of the
  search, or where the three values do not make a peak). In pixels, east and south positive.
- **The interval:** the 90% percentile interval of each axis's estimate over 2,000 bootstrap resamples: exp98's points
  resampled by annotation task (its clusters), exp99's plots resampled within country.

## The sets and what each can show

- **T, exp98's training points on the map (in-sample).** The model was fitted to these labels at these pixels, so its
  map agrees with them best where the code puts them. T tests the record's own pixel-finding code; it says nothing
  about the imagery's geolocation.
- **V, exp98's graded validation points (259).** Held-out points from the same experts: V tests whether the map and
  AWF's labels line up at points the model was not fitted to.
- **S, exp99's plots with 2017 input (271), STRICT primary, LENIENT reported.** Independent labels at 30 m plots,
  reprojected from WGS84: S tests the reprojection and the plots' registration to the map.

## The rule (per set)

- **Aligned:** both axes' 90% intervals lie within -0.5 to +0.5 px: the map agrees best at the labelled pixel itself.
- **Offset:** either axis's interval lies entirely beyond 0.5 px on one side, or the full-sample maximum is on the edge
  of the search (3 px): the map agrees best at a neighbouring pixel, by the estimated amount.
- **Not determined:** anything else.

Expected, stated before the run: T aligned; V aligned or not determined; S not determined is likely, since a 30 m plot
spans three pixels and the map's classes are smooth over a few pixels, which flattens the surface.

## What would change the conclusion

- **T offset:** the record's pixel-finding code is wrong. exp98's graded numbers are void until the cause is traced to
  its commit and every text it reached is corrected.
- **S offset and T aligned:** check exp99's reprojection (axis order, UTM zone, datum) before anything else; if the
  code is right, the offset belongs to the plots' registration and is reported as such.
- **V offset and T aligned:** AWF's labels and the deployed map disagree in position at held-out points by the
  estimated amount (imagery registration or label placement). exp98's agreement at the best position is reported
  beside its graded figure; no verdict changes, since the grades were made at the labelled pixel as frozen.
- **Aligned or not determined everywhere:** no correction; the record says how far a shift could go before the check
  would have seen it (the intervals).

## What it cannot show

- A constant offset between Sentinel-2 and AWF's labels that was already present in the training data could be
  learned by the model and so be invisible in V.
- An offset that differs from point to point (geolocation noise) flattens the surface without moving its maximum: the
  check sees systematic offsets only.
- The map's classes are smooth over a few pixels, so offsets well under a pixel are not resolved.

## Outputs

`exp/out/exp100_summary.json`: per set, the counts, the 7 x 7 mean surfaces of the score and of agreement, the
estimate, the intervals, the call, and agreement at the labelled pixel and at the best position. Offsets are relative,
in pixels; nothing written holds a coordinate, a transform, a bound or a point's identity.

The job, `exp/jobs/e100.sh`, runs on the aicr cluster's `cpu` partition (2 CPUs, 16 GB, 1 hour limit) from a
checkout pinned to a full commit, reads exp98's and exp99's maps and labels where their jobs left them, and writes
only the summary.

## Deviations

- **The commit run.** The page was frozen at 5c4a0ca; the run used 70b2c01, whose only change is exp89's extraction
  dating every extracted file now. The first job (1271993, at 5c4a0ca) stopped before reading any point: the scratch
  purge had removed exp89's AWF label extraction, whose files kept the tar members' old dates. The labels were
  extracted again from the pinned tar (job 1272188) and the check ran on them unchanged.
- **The consequences when T is not determined.** The page states what follows when T is aligned or an offset, and
  writes V's consequence for the case of an aligned T; T came out not determined. The record reads T's intervals (at
  most about 1.1 px) as no sign of a code offset of more than about a pixel, and applies V's frozen consequence (report
  the agreement at the best position beside exp98's verdicts) with T not determined. T's agreement surface is flat
  around the labelled pixel, and its tasks split 8 east and 8 west and lean north (15 north, 3 south).
- **A follow-up after the run.** `exp/exp100_by_task.py` (descriptive, not preregistered, run at 02a1c34, job 1273262) reports each annotation
  task's own best position and V's estimate with each task left out: 13 of V's 20 tasks are best east, 3 west, and the
  estimate stays between 1.3 and 2.1 px east without any one task.

