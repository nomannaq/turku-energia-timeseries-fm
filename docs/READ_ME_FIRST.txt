Since AI Hackathon 2026 - Turku Energia challenge
Imbalance price forecast, 10 hours ahead

Contents
  1_Turku_Energia_Challenge_Brief_EN.pdf      Challenge brief (English translation)
  starter_package_v1.zip                      Data, validation and scoring scripts, baseline
  SHA256SUMS.txt                              Checksums for the starter package

Start here
  1. Read the challenge brief.
  2. Unzip starter_package_v1.zip and read starter_package_v1/README.md.
     It defines the data rule (first_usable_origin_utc), the submission
     format and the scoring.
  3. Check your submission with validate_submission.py before submitting.
     Submissions are made via the Since AI platform.

Data terms
  The Nord Pool spot price data (data/spot_nordpool.csv) is licensed by
  Turku Energia for use during the hackathon only: no redistribution and
  no publication in public repositories. Delete the data after the event.
  Do not commit the data files to a public repository.

Note: starter_package_v1.zip is the locked package of 3 Nov 2026.
Verify it with: sha256sum -c SHA256SUMS.txt --ignore-missing
