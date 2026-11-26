# Evaluation

Measures how often SkillGap agrees with a person's own judgment of fit.

## 1. Add your resume (kept private)
Save your resume as plain text to `eval/data/resume.txt`. The `eval/data/` folder is in
`.gitignore`, so it never reaches GitHub.

## 2. Add postings
Save each job description as `eval/postings/<id>.txt`, for example
`eval/postings/waters-it-intern.txt`. Aim for 20–30, with a real mix: some strong fits,
some okay, some clearly poor.

## 3. Label them *before* scoring
Add a row per posting to `eval/labels.csv`:

```
id,company,role,label,notes
waters-it-intern,Waters,IT Intern - Software Development,good,"Meet all requirements"
```

`label` is your honest judgment, made without looking at SkillGap's score:

- **good**: you meet nearly all the requirements
- **okay**: you meet most requirements but have real gaps
- **poor**: you're missing key requirements

## 4. Run it
```
python eval/run_eval.py
```

The report prints and is saved to `eval/results.md`. It compares three scorers (SkillGap's
full score, skill coverage only, TF-IDF only) on:

- **Pairwise agreement**: of postings you labelled differently, how often the better fit
  scored higher. 50% = coin flip, 100% = perfect.
- **Spearman correlation**: how closely scores rise with your labels (-1 to 1).
- **Label accuracy**: scores turned into labels (good >= 60, okay >= 40).
