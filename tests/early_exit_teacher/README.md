# Early Exit Visualization

## No freezing
This visualization shows the early exit behavior of the model with different KL strengths when sampling from early exiting but not committing to early exits ([visualization](https://htmlpreview.github.io/?https://github.com/MeridianResearch/externalization/blob/karthik/tests/early_exit_teacher/visualizations/unfrozen_teacher_output.html),[code](https://github.com/MeridianResearch/externalization/blob/karthik/tests/early_exit_teacher/unfrozen_teacher.py)) .

## Committing to early exiting
This visualization shows the early exit behavior of the model when sampling from early exiting and committing to early exits ([visualization](https://htmlpreview.github.io/?https://github.com/MeridianResearch/externalization/blob/karthik/tests/early_exit_teacher/visualizations/frozen_teacher_output.html),[code](https://github.com/MeridianResearch/externalization/blob/karthik/tests/early_exit_teacher/frozen_teacher.py)).

The [generation](https://htmlpreview.github.io/?https://github.com/MeridianResearch/externalization/blob/karthik/tests/early_exit_teacher/visualizations/frozen_mlp_teacher_output.html) looks more interesting if we just freeze the MLP operations!! (Code is not pushed yet)
