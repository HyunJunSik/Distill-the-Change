# Reported results and source-run results

The README homogeneous table is transcribed from the manuscript values explicitly
supplied by the repository author on 2026-11-02. Its values and signed gains are
retained as supplied. It is not a table generated from executing this public release.
Teacher/student header accuracies are also author-supplied.

Earlier archived experiment summaries and the later supplied manuscript table
do not have identical baseline/ALTER assignments in every cell. For example,
the archived ResNet56 -> ResNet20 ReviewKD run reports baseline 70.73 and ALTER
70.38, whereas the supplied manuscript table assigns baseline 70.38 and ALTER
70.73. The archived WRN-40-2 -> WRN-16-2 InDistill run likewise reports baseline
75.68 and ALTER 74.97, while the supplied table uses the reverse assignment.

Accordingly, the table is labelled as **author-supplied manuscript results**.
The public release does not reassign raw experiment labels, infer missing run IDs,
claim that these discrepancies were resolved, or guarantee that its new default
commands reproduce every displayed number. Exact run identifiers/checkpoint
mapping for the supplied table remain to be supplied by the authors.

Fresh execution writes actual measured `result.json` values with the objective,
ALTER flag, seed, final epoch and protocol digest. These machine-readable records
retain the signs of any decrease as well as increases. They should remain the
source of any table generated from new executions.

No heterogeneous or ImageNet result table has been added without a request for
those results in this release. No five-run average or standard deviation is inferred.
