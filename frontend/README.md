# CreditLens Frontend

Interactive credit scoring app with SHAP explainability and fairness audit.

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3003

`NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_SITE_URL` are inlined at build time. Under
`npm run dev` they default to the local engine (`http://localhost:8002`) and server;
a production build without them runs the engine's recorded model in the browser
(`lib/engineModel.mjs`), makes no engine call and ships no preview image. `npm test` checks that, and `npm run build` fails if the
prerendered page advertises localhost.

## Features

- Live applicant scoring (22 input fields across 5 sections: every input the model uses;
  "Months Since Last Delinquency" appears once delinquencies > 0)
- Animated score gauge (0–1000), with a risk band and the default rate observed in it on
  the audit set.
- SHAP-style feature contribution waterfall
- Top positive/negative factor lists
- Model-wide fairness audit (demographic parity, equal opportunity, equalized odds,
  disparate impact), with each group's calibration gap and a large-draw note
- Three preset profiles: Strong / Median Applicant / Weak
