# Design-omission share: exclusion test

Delta is the projection of the answer onto the weighted-unweighted
segment (0 = design-correct, 1 = unweighted).

## Protocol: direct
- delta >= 0.7: n=513, of which 87 (17%) land within the task tolerance of the *unweighted* value; 293 mention any weight column
- delta <= 0.3: n=482, of which 9 (2%) land within tolerance of the unweighted value; 263 mention any weight column
- weight mention is not observable in the direct protocol

## Protocol: code
- delta >= 0.7: n=183, of which 124 (68%) land within the task tolerance of the *unweighted* value; 48 mention any weight column
- delta <= 0.3: n=341, of which 40 (12%) land within tolerance of the unweighted value; 313 mention any weight column
- weight-mention rate: 26% (high delta) vs 92% (low delta)

## Protocol: sql
- delta >= 0.7: n=112, of which 47 (42%) land within the task tolerance of the *unweighted* value; 61 mention any weight column
- delta <= 0.3: n=403, of which 55 (14%) land within tolerance of the unweighted value; 389 mention any weight column
- weight-mention rate: 54% (high delta) vs 97% (low delta)

