# XonLearn Customer Conversion Prediction

A project exploring whether a prospective XonLearn learner converts into a paid customer. It compares decision-tree classifiers, reports held-out test metrics, and examines feature importance.

**Author:** [Mahdi Ataeifar](https://github.com/mahdiataeifar)

## Project contents

- `XonLearn.csv` — the lead/customer-interaction dataset (4,612 rows, 15 columns).

## Dataset

Each row represents a prospective customer. The columns are:

| Column | Meaning |
| --- | --- |
| `ID` | Row/customer identifier; excluded from predictors. |
| `age` | Age. |
| `current_occupation` | Occupation category (e.g. Professional, Unemployed, Student). |
| `first_interaction` | First contact channel (Website or Mobile App). |
| `profile_completed` | Profile completion band (Low, Medium, High). |
| `website_visits` | Number of website visits. |
| `time_spent_on_website` | Website time, in seconds. |
| `page_views_per_visit` | Average page views per visit. |
| `last_activity` | Last recorded interaction category (Email, Phone, or Website Activity). |
| `print_media_type1`, `print_media_type2` | Print-ad exposure flags. |
| `digital_media` | Digital-ad exposure flag. |
| `educational_channels` | Exposure through educational channels. |
| `referral` | Referral exposure flag. |
| `status` | Target: `1` = converted/paid customer; `0` = not converted. |

The target is imbalanced: approximately 70.1% class 0 and 29.9% class 1. The CSV provides the actual category values; the model pipeline one-hot encodes categorical predictors rather than imposing an artificial category order.

## Method

1. Explore class balance and selected feature relationships with plots and descriptive summaries.
2. Set `X` to all columns except `ID` and `status`; use `status` as `y`.
3. Make one stratified 70%/30% train/test split (`random_state=0`). This gives 3,228 training and 1,384 test rows, preserving the class ratio (about 70/30 in each).
4. Fit preprocessing as part of each scikit-learn pipeline: numeric columns are median-imputed; categorical columns are most-frequent-imputed and one-hot encoded with `handle_unknown='ignore'`. Since the transformers are fit within the pipeline on training data (and inside CV folds for the search), test-fold preprocessing statistics are not used for fitting.
5. Compare a `DecisionTreeClassifier(max_depth=7, random_state=0)` and a `RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)` on the same held-out test set.
6. Additionally, search decision-tree depth over `[3, 5, 7, 10, None]` using `GridSearchCV`, stratified 5-fold CV, and class-1 F1 on the training portion only. The selected depth is 7 (mean CV F1 = 0.7507). This is a small, explicitly limited grid—not an exhaustive search. The selected tree is evaluated on the same untouched test set.

## Held-out test results

Metrics below are for the positive/conversion class (`status=1`), except accuracy. They are from the notebook's test predictions (1,384 examples); values are rounded to three decimals.

| Model | Accuracy | Precision (class 1) | Recall (class 1) | F1 (class 1) |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree (depth 7) | 0.844 | 0.749 | 0.717 | 0.733 |
| Random Forest (300 trees) | 0.849 | 0.767 | 0.709 | 0.737 |
| Tuned Decision Tree (5-fold CV; depth 7) | 0.844 | 0.749 | 0.717 | 0.733 |

The Random Forest has the highest class-1 F1 among the two baseline models in this run (0.737 versus 0.733), and slightly higher accuracy and precision; the Decision Tree has slightly higher class-1 recall. The tuned tree has the same test scores as the depth-7 baseline tree. These are results from a single stratified holdout split, not estimates across repeated independent test sets.

### Feature-importance observation

In the fitted Random Forest, impurity-based importances were summed across one-hot categories back to their original columns. The largest reported values were `time_spent_on_website` (0.2469), `first_interaction` (0.1760), `profile_completed` (0.1182), `page_views_per_visit` (0.1103), and `age` (0.1037). These are model-specific predictive importances, not causal effects, and can be affected by correlated predictors and feature-importance bias. The exploratory plots likewise show associations, not causes.

## Reproducibility and limitations

- The split and model random states are fixed at 0; the CV shuffling seed is also 0. Package/version changes can still affect results.
- The notebook contains recorded outputs. Re-running the cells is necessary to independently reproduce the metrics; the README's table reflects those stored outputs and their confusion matrices.
- The conversion class is the minority class, so accuracy alone is insufficient; precision, recall, and F1 are reported for class 1.
- Performance is based on this dataset and a single holdout split. Feature importances are not causal explanations.

