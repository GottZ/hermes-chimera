# Third-party notices

chimera is distributed under the MIT License (see `LICENSE`). It contains material from the projects below; their
notices follow.

## chimera-vocal-synth (MIT)

The synthesis engine in `chimera_vocal/` is a Python port of the Rust engine of chimera-vocal-synth by Jameson Hodge,
<https://github.com/jamesondh/chimera-vocal-synth>, commit `092b16ce12b927d473c884ca2dfae53ce1b995df` (2026-11-08).
The test references under `tests/golden/` and `tests/vectors/` were copied or exported from the same commit; their
README files list each file with its sha256.

Data files taken over unchanged:

| file in this repository | source in chimera-vocal-synth | bytes | sha256 |
|---|---|---|---|
| `chimera_vocal/data/cmudict.txt` | `public/data/cmudict.txt.gz` (decompressed) | 3330130 | `cac51a2d86b7312240d4c16b52d9c606a6989d8e03ced24615b2b573510a01af` |
| `chimera_vocal/data/vowel_targets.json` | `crates/chimera/src/phonetics/vowelTargets.generated.json` | 25987 | `975ab3fec65772b6c232fea732808147390e0da9d6240dbffdea4b97a557235d` |
| `chimera_vocal/data/consonant_targets.json` | `crates/chimera/src/phonetics/consonantTargets.generated.json` | 1907 | `aac1d0868d0af07e39e482a05c1e27f0bb628d5640e54231aeede162d0a8f17d` |

```
MIT License

Copyright (c) 2026 Jameson Hodge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## CMUdict (BSD-style licence)

`chimera_vocal/data/cmudict.txt` is the Carnegie Mellon University Pronouncing Dictionary as shipped by
chimera-vocal-synth (`public/data/cmudict.txt.gz`, licence file `public/data/CMUDICT-LICENSE.txt`). The English
pronunciations in `tests/golden/phonetics-*.json` were produced from it. Licence text, verbatim:

```
Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   The contents of this file are deemed to be source code.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.

This work was supported in part by funding from the Defense Advanced
Research Projects Agency, the Office of Naval Research and the National
Science Foundation of the United States of America, and by member
companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge
the contributions of many volunteers to the expansion and improvement of
this dictionary.

THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
